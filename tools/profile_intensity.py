#!/usr/bin/env python3
"""
Arithmetic-Intensity (Roofline) Profiler for AccelMark.

Loads a model directly via transformers/torch and profiles two phases
separately at the suite's operating point:

  - **prefill** : one forward pass over ``--prompt-len`` tokens at ``--batch``.
  - **decode**  : one single-token step with a KV cache at ``--batch``.

Outputs a JSON file classifying each phase as compute-bound or
bandwidth-bound relative to the chip's roofline ridge point.

Usage::

    python tools/profile_intensity.py \\
        --suite suite_A \\
        --model meta-llama/Llama-3-8B \\
        --batch 32 --prompt-len 280 --gen-len 128 \\
        --dtype bfloat16 \\
        --out results/profiling/A100_suite_A_intensity.json

Defaults for ``--batch``, ``--prompt-len``, ``--gen-len``, and ``--dtype``
are read from the suite JSON so the operating point matches the benchmark;
CLI flags override.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

# Repo root — tools/profile_intensity.py is one level down from root
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ── Known chip peak specs ────────────────────────────────────────────────────
# FP16/BF16 tensor-core peak TFLOPS and HBM bandwidth in GB/s.
# Sources: vendor datasheets. Used when --peak-tflops/--peak-bw-gbps are not
# passed on the CLI. Contributions for new chips welcome.

_CHIP_PEAK_SPECS: dict[str, dict[str, float]] = {
    "NVIDIA A100-SXM4-80GB":      {"tflops": 312.0, "bw_gbps": 2039.0},
    "NVIDIA A100-SXM4-40GB":      {"tflops": 312.0, "bw_gbps": 1555.0},
    "NVIDIA A100-PCIe-80GB":      {"tflops": 312.0, "bw_gbps": 1935.0},
    "NVIDIA H100-SXM-80GB":       {"tflops": 989.0, "bw_gbps": 3350.0},
    "NVIDIA H100-PCIe-80GB":      {"tflops": 756.0, "bw_gbps": 2039.0},
    "NVIDIA H200-SXM-141GB":      {"tflops": 989.0, "bw_gbps": 4800.0},
    "NVIDIA H20-3e":              {"tflops": 148.0, "bw_gbps": 4000.0},
    "NVIDIA RTX 5090":            {"tflops": 104.8, "bw_gbps": 1790.0},
    "NVIDIA L40S":                {"tflops": 362.0, "bw_gbps": 864.0},
    "NVIDIA V100-SXM2-32GB":      {"tflops": 125.0, "bw_gbps": 900.0},
    "NVIDIA T4":                  {"tflops": 65.0,  "bw_gbps": 320.0},
}


def _detect_chip_name() -> str:
    """Best-effort detection of the current GPU name via PyTorch or nvidia-smi."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).name
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.strip().splitlines()[0].strip()
    except Exception:
        pass
    return "unknown"


def _resolve_peak(chip: str, cli_tflops: float | None, cli_bw: float | None):
    """Return ``(tflops, bw_gbps, source)`` for the given chip."""
    if cli_tflops is not None and cli_bw is not None:
        return cli_tflops, cli_bw, "cli-flags"

    spec = _CHIP_PEAK_SPECS.get(chip)
    if spec:
        return spec["tflops"], spec["bw_gbps"], f"builtin:{chip}"

    return None, None, "unknown"


def _ridge_point(tflops: float, bw_gbps: float) -> float:
    """Roofline ridge point I* = peak FLOP/s / peak byte/s."""
    return round((tflops * 1e12) / (bw_gbps * 1e9), 2)


def _classify_phase(arithmetic_intensity: float, ridge: float) -> str:
    """Classify a phase as compute-bound or bandwidth-bound."""
    return "compute-bound" if arithmetic_intensity > ridge else "bandwidth-bound"


def _parse_dtype(dtype_str: str):
    """Parse a dtype string to a torch dtype and canonical name."""
    import torch
    _map = {
        "bfloat16": (torch.bfloat16, "bfloat16"),
        "bf16":     (torch.bfloat16, "bfloat16"),
        "float16":  (torch.float16, "float16"),
        "fp16":     (torch.float16, "float16"),
        "float32":  (torch.float32, "float32"),
        "fp32":     (torch.float32, "float32"),
    }
    key = dtype_str.lower().strip()
    if key in _map:
        return _map[key]
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _model_bytes(param_count: int, dtype_str: str) -> int:
    """Estimated weight footprint in bytes for a given dtype."""
    bytes_per_param = 1 if "int8" in dtype_str else (4 if "float32" in dtype_str else 2)
    return param_count * bytes_per_param


def _kv_cache_bytes(model_config, batch: int, seq_len: int, dtype_str: str) -> int:
    """Estimated KV-cache footprint in bytes for one full sequence."""
    num_layers = getattr(model_config, "num_hidden_layers", 0)
    num_kv_heads = getattr(
        model_config, "num_key_value_heads",
        getattr(model_config, "num_attention_heads", 0),
    )
    head_dim = (
        getattr(model_config, "hidden_size", 0)
        // getattr(model_config, "num_attention_heads", 1)
    )
    bytes_per_elem = 1 if "int8" in dtype_str else (4 if "float32" in dtype_str else 2)
    # 2 × for K and V
    return 2 * num_layers * batch * seq_len * num_kv_heads * head_dim * bytes_per_elem


# ── Profiling helpers ────────────────────────────────────────────────────────

def _import_flop_counter():
    """Return ``FlopCounterMode`` or None if unavailable (older PyTorch)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        return FlopCounterMode
    except ImportError:
        return None


def _analytical_prefill_flops(param_count: int, batch: int, seq_len: int, model_config) -> int:
    """Approximate FLOPs for a single prefill forward pass.

    Uses the standard 2 × params × tokens approximation for dense transformers,
    which captures matmul and attention FLOPs at large batch sizes.
    """
    return 2 * param_count * batch * seq_len


def _analytical_decode_flops(param_count: int, batch: int, model_config) -> int:
    """Approximate FLOPs for a single decode step (one token per sequence)."""
    return 2 * param_count * batch


def _cuda_timed_block(model_fn, warmup: int = 5, trials: int = 5):
    """Time ``model_fn()`` over ``trials`` iterations after ``warmup`` warmup
    passes. Returns ``(median_time_ms, flops)`` where *flops* is from the
    last trial's ``FlopCounterMode`` (or None if unavailable).

    All CUDA work is fenced with ``torch.cuda.Event`` for accurate GPU timing.
    """
    import torch

    FlopCounterMode = _import_flop_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    times_ms = []
    last_flops = None

    for i in range(warmup + trials):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        flop_ctx = FlopCounterMode(display=False) if FlopCounterMode else None
        if device == "cuda":
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)

        with (flop_ctx if flop_ctx else _NullContext()):
            if device == "cuda":
                start_ev.record()
            with torch.no_grad():
                model_fn()
            if device == "cuda":
                end_ev.record()
                torch.cuda.synchronize()

        if device == "cuda":
            elapsed = start_ev.elapsed_time(end_ev)
        else:
            elapsed = 0.0

        if i >= warmup:
            times_ms.append(elapsed)
        if flop_ctx:
            last_flops = flop_ctx.get_total_flops()

    median_ms = sorted(times_ms)[len(times_ms) // 2] if times_ms else 0.0
    return median_ms, last_flops


class _NullContext:
    """No-op context manager for when FlopCounterMode is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ── Main profiler ────────────────────────────────────────────────────────────

def profile(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: No CUDA device detected; timing will not be meaningful.")

    chip = _detect_chip_name()
    dtype_str = args.dtype

    torch_dtype, canonical_dtype = _parse_dtype(dtype_str)
    try:
        _ = torch.tensor([1.0], dtype=torch_dtype, device=device)
    except RuntimeError:
        # dtype not supported on this device — fall back to float16
        print(f"WARNING: {dtype_str} not supported on {device}, falling back to float16")
        torch_dtype, canonical_dtype = torch.float16, "float16"

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading {args.model} ({canonical_dtype}) on {chip} ...")
    t_load = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  {param_count / 1e9:.1f}B params loaded in {time.perf_counter() - t_load:.1f}s")

    batch      = args.batch
    prompt_len = args.prompt_len
    gen_len    = args.gen_len

    # ── Phase 1: Prefill ───────────────────────────────────────────────────
    print(f"\nProfiling prefill — batch={batch}, prompt_len={prompt_len}")

    torch.manual_seed(42)
    input_ids = torch.randint(0, model.config.vocab_size, (batch, prompt_len), device=device)
    attn_mask = torch.ones(batch, prompt_len, device=device)

    def _prefill_step():
        return model(input_ids=input_ids, attention_mask=attn_mask)

    # Warmup-only pass to JIT-compile
    with torch.no_grad():
        _prefill_step()
    if device == "cuda":
        torch.cuda.synchronize()

    prefill_ms, prefill_flops = _cuda_timed_block(_prefill_step)

    # FLOPs fallback: analytical approximation
    flops_method = "measured" if prefill_flops is not None else "analytical"
    if prefill_flops is None:
        prefill_flops = _analytical_prefill_flops(param_count, batch, prompt_len, model.config)

    # Bytes modelled as: weight footprint + KV cache
    weight_bytes = _model_bytes(param_count, canonical_dtype)
    kv_bytes = _kv_cache_bytes(model.config, batch, prompt_len, canonical_dtype)
    prefill_bytes = weight_bytes + kv_bytes  # each token loads all weights once

    prefill_tflops = (prefill_flops / (prefill_ms / 1000)) / 1e12 if prefill_ms > 0 else 0.0
    prefill_bw_gbps = (prefill_bytes / (prefill_ms / 1000)) / 1e9 if prefill_ms > 0 else 0.0
    prefill_ai = round(prefill_flops / prefill_bytes, 2) if prefill_bytes > 0 else 0.0

    print(f"  FLOPs={prefill_flops:.2e}  time={prefill_ms:.1f}ms  "
          f"TFLOPS={prefill_tflops:.1f}  AI={prefill_ai:.1f} FLOP/byte")

    # ── Phase 2: Decode ───────────────────────────────────────────────────
    print(f"Profiling decode — batch={batch}, single-token step with KV cache")

    # Prime the KV cache with a full prefill
    with torch.no_grad():
        full_out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
        past_key_values = full_out.past_key_values

    # Single-token decode input
    decode_ids = input_ids[:, -1:].contiguous()  # shape (batch, 1)
    decode_mask = torch.ones(batch, prompt_len + 1, device=device)

    def _decode_step():
        return model(
            input_ids=decode_ids,
            attention_mask=decode_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

    # Warmup decode
    with torch.no_grad():
        _decode_step()
    if device == "cuda":
        torch.cuda.synchronize()

    decode_ms, decode_flops = _cuda_timed_block(_decode_step, warmup=5, trials=15)

    if decode_flops is None:
        decode_flops = _analytical_decode_flops(param_count, batch, model.config)

    # Decode bytes: weight footprint + KV cache read
    decode_bytes = weight_bytes + kv_bytes

    decode_tflops = (decode_flops / (decode_ms / 1000)) / 1e12 if decode_ms > 0 else 0.0
    decode_bw_gbps = (decode_bytes / (decode_ms / 1000)) / 1e9 if decode_ms > 0 else 0.0
    decode_ai = round(decode_flops / decode_bytes, 2) if decode_bytes > 0 else 0.0

    print(f"  FLOPs={decode_flops:.2e}  time={decode_ms:.1f}ms  "
          f"TFLOPS={decode_tflops:.1f}  AI={decode_ai:.1f} FLOP/byte")

    # ── Roofline classification ────────────────────────────────────────────
    peak_tflops, peak_bw, peak_source = _resolve_peak(chip, args.peak_tflops, args.peak_bw_gbps)

    if peak_tflops is None or peak_bw is None:
        print("WARNING: Chip peak specs unknown. Pass --peak-tflops and --peak-bw-gbps.")
        ridge = None
        prefill_class = "unknown"
        decode_class  = "unknown"
    else:
        ridge = _ridge_point(peak_tflops, peak_bw)
        prefill_class = _classify_phase(prefill_ai, ridge)
        decode_class  = _classify_phase(decode_ai, ridge)
        print(f"\nRoofline ridge I* = {ridge:.1f} FLOP/byte  (source: {peak_source})")
        print(f"  Prefill: I={prefill_ai:.1f}  →  {prefill_class}")
        print(f"  Decode:  I={decode_ai:.1f}  →  {decode_class}")

    return {
        "suite": args.suite,
        "chip": chip,
        "model": args.model,
        "operating_point": {
            "batch": batch,
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "dtype": canonical_dtype,
        },
        "phases": {
            "prefill": {
                "flops": prefill_flops,
                "bytes": prefill_bytes,
                "arithmetic_intensity": prefill_ai,
                "achieved_tflops": round(prefill_tflops, 2),
                "achieved_bw_gbps": round(prefill_bw_gbps, 2),
                "time_ms_median": round(prefill_ms, 2),
                "flops_method": flops_method,
                "bytes_method": "modelled",
            },
            "decode": {
                "flops": decode_flops,
                "bytes": decode_bytes,
                "arithmetic_intensity": decode_ai,
                "achieved_tflops": round(decode_tflops, 2),
                "achieved_bw_gbps": round(decode_bw_gbps, 2),
                "time_ms_median": round(decode_ms, 2),
                "flops_method": flops_method,
                "bytes_method": "modelled",
            },
        },
        "chip_peak": {
            "tflops": peak_tflops,
            "bw_gbps": peak_bw,
            "ridge_point_i_star": ridge,
            "source": peak_source,
        },
        "classification": {
            "prefill": prefill_class,
            "decode": decode_class,
        },
        "env_ref": chip,
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def _load_suite_defaults(suite_id: str) -> dict:
    """Read default operating point from a suite's suite.json."""
    suite_path = _REPO_ROOT / "suites" / suite_id / "suite.json"
    if not suite_path.exists():
        return {}
    with open(suite_path) as f:
        s = json.load(f)
    dist = s.get("request_distribution", {})
    defaults = {}
    if dist.get("input_tokens_p50"):
        defaults["prompt_len"] = dist["input_tokens_p50"]
    if dist.get("output_tokens_p50"):
        defaults["gen_len"] = dist["output_tokens_p50"]
    prec = s.get("precision_required", "bfloat16").lower()
    defaults["dtype"] = "bfloat16" if prec == "bf16" else prec
    return defaults


def main():
    parser = argparse.ArgumentParser(
        description="AccelMark Arithmetic-Intensity (Roofline) Profiler"
    )
    parser.add_argument("--suite", required=True,
                        help="Suite ID (e.g. suite_A)")
    parser.add_argument("--model", default=None,
                        help="Model ID (default: from suite.json)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--prompt-len", type=int, default=None,
                        help="Prompt length in tokens")
    parser.add_argument("--gen-len", type=int, default=None,
                        help="Generation length (informational)")
    parser.add_argument("--dtype", default=None,
                        help="Compute dtype: bfloat16, float16, float32")
    parser.add_argument("--peak-tflops", type=float, default=None,
                        help="Chip peak FP16/BF16 TFLOPS")
    parser.add_argument("--peak-bw-gbps", type=float, default=None,
                        help="Chip peak HBM bandwidth in GB/s")
    parser.add_argument("--out", default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    # Apply suite defaults for unspecified arguments
    defaults = _load_suite_defaults(args.suite)
    if args.model is None:
        suite_path = _REPO_ROOT / "suites" / args.suite / "suite.json"
        if suite_path.exists():
            with open(suite_path) as f:
                args.model = json.load(f).get("model_id")
    if args.model is None:
        parser.error("--model is required (could not resolve from suite.json)")
    for attr in ("batch", "prompt_len", "gen_len", "dtype"):
        if getattr(args, attr) is None:
            setattr(args, attr, defaults.get(attr))
    if args.batch is None:
        args.batch = 32
    if args.prompt_len is None:
        args.prompt_len = 280
    if args.gen_len is None:
        args.gen_len = 128
    if args.dtype is None:
        args.dtype = "bfloat16"

    result = profile(args)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nOutput written to {out_path}")
    else:
        print("\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
