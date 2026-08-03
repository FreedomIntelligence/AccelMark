#!/usr/bin/env python3
"""
Arithmetic-Intensity (Roofline) Profiler for AccelMark.

Loads a model directly via transformers / torch and profiles two phases
separately at the suite's operating point:

  - **prefill** : one forward pass over ``--prompt-len`` tokens at ``--batch``.
  - **decode**  : one single-token step with a KV cache at ``--batch``.

Uses platform-specific profiling backends (NVIDIA CUDA, Huawei Ascend, etc.)
to obtain hardware-measured FLOPs and DRAM traffic when available, falling
back to analytical / modelled estimates otherwise.

Outputs a JSON file classifying each phase as compute-bound or
bandwidth-bound relative to the chip's roofline ridge point.

Usage::

    python tools/profile_intensity.py \\
        --suite suite_A \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --batch 32 --prompt-len 280 --gen-len 128 \\
        --dtype bfloat16 \\
        --out results/profiling/A800_suite_A_intensity.json

Defaults for ``--batch``, ``--prompt-len``, ``--gen-len``, and ``--dtype``
are read from the suite JSON so the operating point matches the benchmark;
CLI flags override.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root — tools/profile_intensity.py is one level down from root
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from profiling import IntensityProfiler, get_backend_by_id, list_known_chips


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
        defaults["output_tokens_p50"] = dist["output_tokens_p50"]
    prec = s.get("precision_required", "bfloat16").lower()
    defaults["dtype"] = "bfloat16" if prec == "bf16" else prec
    return defaults


def _resolve_model_id(suite_id: str) -> str | None:
    """Read model_id from suite.json."""
    suite_path = _REPO_ROOT / "suites" / suite_id / "suite.json"
    if not suite_path.exists():
        return None
    with open(suite_path) as f:
        return json.load(f).get("model_id")


def main():
    parser = argparse.ArgumentParser(
        description="AccelMark Arithmetic-Intensity (Roofline) Profiler"
    )
    parser.add_argument(
        "--suite", default=None,
        help="Suite ID (e.g. suite_A) — required for profiling, "
             "not needed with --list-chips / --list-backends")
    parser.add_argument(
        "--model", default=None,
        help="Model ID (default: from suite.json)")
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Batch size")
    parser.add_argument(
        "--prompt-len", type=int, default=None,
        help="Prompt length in tokens")
    parser.add_argument(
        "--gen-len", type=int, default=None,
        help="Generation length (informational)")
    parser.add_argument(
        "--dtype", default=None,
        help="Compute dtype: bfloat16, float16, float32")
    parser.add_argument(
        "--peak-tflops", type=float, default=None,
        help="Chip peak FP16/BF16 TFLOPS")
    parser.add_argument(
        "--peak-bw-gbps", type=float, default=None,
        help="Chip peak HBM bandwidth in GB/s")
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path")
    parser.add_argument(
        "--env-file", default=None,
        help="Path to env_info.json for provenance tracking")
    parser.add_argument(
        "--backend", default=None,
        help="Force a specific profiling backend (e.g. nvidia, ascend). "
             "Default: auto-detect.")
    parser.add_argument(
        "--list-chips", action="store_true",
        help="List known chip peak specs and exit.")
    parser.add_argument(
        "--list-backends", action="store_true",
        help="List available profiling backends and exit.")
    parser.add_argument(
        "--list-i-star", action="store_true",
        help="Print I* (ridge-point) table for all known chips and exit.")
    args = parser.parse_args()

    # ── List modes ────────────────────────────────────────────────────────
    if args.list_chips:
        from profiling import lookup
        print("Known chip peak specs:\n")
        print(f"{'Chip':45s} {'TFLOPS':>8s} {'BW GB/s':>8s}")
        print("-" * 64)
        for name in list_known_chips():
            tflops, bw = lookup(name)
            t_str = f"{tflops:.0f}" if tflops else "?"
            b_str = f"{bw:.0f}" if bw else "?"
            print(f"  {name:42s} {t_str:>8s} {b_str:>8s}")
        return

    if args.list_backends:
        from profiling import discover_backends
        print("Available profiling backends:\n")
        for b in discover_backends():
            available = b.is_available()
            status = "✓ detected" if available else "✗ not available"
            print(f"  {b.ID:12s}  PRIORITY={b.PRIORITY:<3d}  {status}")
            print(f"  {'':12s}  {b.DISPLAY_NAME}")
            print()
        return

    if args.list_i_star:
        from profiling import print_i_star_table
        print_i_star_table()
        return

    # ── Validate suite is provided for profiling mode ────────────────────
    if not args.suite:
        parser.error("--suite is required for profiling (e.g. --suite suite_A)")

    # ── Resolve defaults from suite ───────────────────────────────────────
    defaults = _load_suite_defaults(args.suite)

    if args.model is None:
        args.model = _resolve_model_id(args.suite)
    if args.model is None:
        parser.error("--model is required (could not resolve from suite.json)")

    for attr in ("batch", "prompt_len", "gen_len", "dtype", "output_tokens_p50"):
        if getattr(args, attr, None) is None:
            setattr(args, attr, defaults.get(attr))
    if args.batch is None:
        args.batch = 32
    if args.prompt_len is None:
        args.prompt_len = 280
    if args.gen_len is None:
        args.gen_len = 128
    if args.dtype is None:
        args.dtype = "bfloat16"

    # ── Resolve backend ───────────────────────────────────────────────────
    backend = None
    if args.backend:
        backend = get_backend_by_id(args.backend)
        if backend is None:
            parser.error(f"Unknown backend: '{args.backend}'. "
                         f"Use --list-backends to see available backends.")

    # ── Profile ───────────────────────────────────────────────────────────
    # Use the canonical HF model ID from suite.json as the output label
    # (--model may be a local path for loading; the output should be portable).
    canonical_model_id = _resolve_model_id(args.suite) or args.model

    profiler = IntensityProfiler(
        suite=args.suite,
        model_id=args.model,
        batch=args.batch,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        dtype_str=args.dtype,
        backend=backend,
        peak_tflops=args.peak_tflops,
        peak_bw_gbps=args.peak_bw_gbps,
        env_file=args.env_file,
        output_tokens_p50=getattr(args, "output_tokens_p50", None),
        model_label=canonical_model_id,
    )

    result = profiler.run()

    # ── Output ────────────────────────────────────────────────────────────
    output_dict = result.to_dict()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"\nOutput written to {out_path}")
    else:
        print("\n" + json.dumps(output_dict, indent=2))


if __name__ == "__main__":
    main()
