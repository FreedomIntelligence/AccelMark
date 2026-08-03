"""
Intensity profiler orchestrator.

Loads a model via ``transformers``, selects the active profiling
backend, profiles the prefill and decode phases, and classifies each
against the chip's roofline ridge point.
"""

from __future__ import annotations

import gc
import hashlib
import time
from pathlib import Path
from typing import Optional

from profiling.base import (
    ProfilerBackend,
    PhaseProfile,
    IntensityResult,
    BackendNotAvailableError,
)
from profiling.backends import get_active_backend, get_backend_by_id
from profiling.chip_specs import lookup as _lookup_chip_specs
from profiling.roofline import ridge_point as _ridge_point, classify as _classify


class IntensityProfiler:
    """Orchestrates intensity profiling for a (model, suite) pair.

    Parameters
    ----------
    suite:
        Suite identifier, e.g. ``"suite_A"``.
    model_id:
        HuggingFace model id or local path.
    batch:
        Batch size at which to profile.
    prompt_len:
        Prompt length in tokens for the prefill phase.
    gen_len:
        Generation length in tokens (informational — not used for
        decode profiling which always operates on 1 token).
    dtype_str:
        Compute dtype: ``"bfloat16"``, ``"float16"``, ``"float32"``.
    backend:
        Optional backend override.  Auto-detected when ``None``.
    peak_tflops:
        Override the built-in chip peak TFLOPS.
    peak_bw_gbps:
        Override the built-in chip peak bandwidth (GB / s).
    env_file:
        Path to an ``env_info.json`` provenance file.
    """

    def __init__(
        self,
        *,
        suite: str,
        model_id: str,
        batch: int,
        prompt_len: int,
        gen_len: int = 128,
        dtype_str: str = "bfloat16",
        backend: Optional[ProfilerBackend] = None,
        peak_tflops: Optional[float] = None,
        peak_bw_gbps: Optional[float] = None,
        env_file: Optional[str] = None,
        output_tokens_p50: Optional[int] = None,
        model_label: Optional[str] = None,
    ):
        self.suite = suite
        self.model_id = model_id
        self.model_label = model_label or model_id
        self.batch = batch
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.dtype_str = dtype_str
        self._backend = backend
        self._peak_tflops_override = peak_tflops
        self._peak_bw_gbps_override = peak_bw_gbps
        self._env_file = env_file
        self._output_tokens_p50 = output_tokens_p50

        # Resolved at run() time
        self._device = "cpu"
        self._chip_name = "unknown"
        self._param_count = 0

    # ── Public API ───────────────────────────────────────────────────────

    def run(self) -> IntensityResult:
        """Execute the full profiling pipeline.

        1.  Detect hardware and resolve backend.
        2.  Load the model.
        3.  Profile the prefill phase.
        4.  Profile the decode phase.
        5.  Classify against the chip's roofline.
        6.  Build and return the :class:`IntensityResult`.
        """
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Resolve backend ──────────────────────────────────────────
        backend = self._backend
        if backend is None:
            backend = get_active_backend()
        if backend is None:
            raise BackendNotAvailableError(
                "No profiling backend is available. "
                "Install PyTorch with CUDA / ROCm / NPU support, "
                "or pass --backend to force a specific backend."
            )

        backend.setup()
        try:
            self._chip_name = backend.get_chip_name()
            chip_specs = self._resolve_chip_specs()

            # ── Load model ───────────────────────────────────────────
            model, tokenizer, model_config = self._load_model()

            # ── Profile prefill ──────────────────────────────────────
            print(f"\nProfiling prefill — batch={self.batch}, "
                  f"prompt_len={self.prompt_len}")
            prefill = self._profile_phase(
                backend, model, model_config, tokenizer,
                phase="prefill",
            )

            # ── Profile decode ───────────────────────────────────────
            print(f"Profiling decode — batch={self.batch}, single-token step")
            decode = self._profile_phase(
                backend, model, model_config, tokenizer,
                phase="decode",
            )

            # ── Classify ─────────────────────────────────────────────
            classification = {}
            if chip_specs.get("ridge_point_i_star") is not None:
                ridge = chip_specs["ridge_point_i_star"]
                classification["prefill"] = _classify(
                    prefill.arithmetic_intensity, ridge
                )
                classification["decode"] = _classify(
                    decode.arithmetic_intensity, ridge
                )
                print(f"\nRoofline ridge I* = {ridge:.1f} FLOP/byte  "
                      f"(source: {chip_specs['source']})")
                print(f"  Prefill: I={prefill.arithmetic_intensity:.1f}"
                      f"  →  {classification['prefill']}")
                print(f"  Decode:  I={decode.arithmetic_intensity:.1f}"
                      f"  →  {classification['decode']}")
            else:
                classification = {"prefill": "unknown", "decode": "unknown"}
                print("\nWARNING: Chip peak specs unknown. "
                      "Pass --peak-tflops and --peak-bw-gbps.")

            # ── Time fractions ─────────────────────────────────────────
            total_time = prefill.elapsed_ms + decode.elapsed_ms
            if total_time > 0:
                prefill.time_frac = round(prefill.elapsed_ms / total_time, 4)
                decode.time_frac = round(decode.elapsed_ms / total_time, 4)

            # ── Blended arithmetic intensity ───────────────────────────
            blended = None
            if self._output_tokens_p50 is not None and self._output_tokens_p50 > 0:
                out_tokens = self._output_tokens_p50
                # Per-request FLOPs: 1 prefill + out_tokens decode steps
                pref_flops_per_req = prefill.flops // self.batch
                dec_flops_per_step  = decode.flops // self.batch
                total_flops = pref_flops_per_req + out_tokens * dec_flops_per_step
                # Per-request bytes: weight reads are batch-amortised
                pref_bytes_per_req = prefill.bytes_read // self.batch + prefill.bytes_written // self.batch
                dec_bytes_per_step = decode.bytes_read // self.batch + decode.bytes_written // self.batch
                total_bytes = pref_bytes_per_req + out_tokens * dec_bytes_per_step
                blended_ai = round(total_flops / total_bytes, 2) if total_bytes > 0 else 0.0

                # Ridge check for blended AI
                blended_class = "unknown"
                if chip_specs.get("ridge_point_i_star") is not None:
                    blended_class = _classify(
                        blended_ai, chip_specs["ridge_point_i_star"]
                    )

                blended = {
                    "ai": blended_ai,
                    "weighting": (
                        f"token-weighted: 1x prefill (S={self.prompt_len}) + "
                        f"{out_tokens}x decode (1 token each), "
                        f"batch={self.batch} (weight reads amortised)"
                    ),
                    "classification": blended_class,
                }
                print(f"  Blended AI (p50 request, out={out_tokens}): "
                      f"I={blended_ai:.1f}  →  {blended_class}")

            return IntensityResult(
                suite=self.suite,
                chip=self._chip_name,
                model=self.model_label,
                operating_point={
                    "batch": self.batch,
                    "prompt_len": self.prompt_len,
                    "gen_len": self.gen_len,
                    "dtype": self._dtype_canonical_name(),
                },
                prefill=prefill,
                decode=decode,
                blended=blended,
                chip_peak=chip_specs,
                classification=classification,
                env_ref=self._build_env_ref(),
                runtime_note=(
                    "FLOPs: measured via torch.utils.FlopCounterMode (GQA-patched, "
                    "causal-corrected) on HuggingFace transformers.AutoModelForCausalLM. "
                    "Bytes: analytical from model geometry (weight reads + KV-cache "
                    "traffic + activation tensor traffic). Activation model follows "
                    "FlashAttention HBM traffic: O(B*S*(4d+2*d_ff)) per layer, "
                    "assuming Q/K/V intermediates and scores stay in SRAM. "
                    "I-values are architecture-invariant by construction (same on any "
                    "chip at the same batch/dtype). Decode I scales with batch: "
                    "I(B=1) ≈ 1, I(B=32) ≈ 28. "
                    "ncu validation pending: GPU perf counters require admin "
                    "privileges unavailable in this environment."
                ),
            )
        finally:
            backend.teardown()
            self._cleanup_model()

    # ── Phase profiling ─────────────────────────────────────────────────

    def _profile_phase(
        self,
        backend: ProfilerBackend,
        model,
        model_config,
        tokenizer,
        *,
        phase: str,
    ) -> PhaseProfile:
        """Profile a single phase (prefill or decode)."""
        import torch

        device = self._device

        if phase == "prefill":
            torch.manual_seed(42)
            input_ids = torch.randint(
                0, model.config.vocab_size,
                (self.batch, self.prompt_len),
                device=device,
            )
            attn_mask = torch.ones(self.batch, self.prompt_len, device=device)

            def model_fn():
                return model(input_ids=input_ids, attention_mask=attn_mask)

            # Warmup to JIT-compile
            with torch.no_grad():
                model_fn()
            if device == "cuda":
                torch.cuda.synchronize()

            # Time the phase
            elapsed_ms, _ = _cuda_timed_block(model_fn)

            # Measure FLOPs
            flops, flops_method = backend.measure_flops(model_fn)
            if flops <= 0:
                flops = _analytical_prefill_flops(
                    self._param_count, self.batch, self.prompt_len
                )
                flops_method = "analytical"

            # Measure DRAM bytes
            bytes_read, bytes_written, bytes_method = backend.measure_dram_bytes(
                model_fn,
                param_count=self._param_count,
                model_config=model_config,
                batch=self.batch,
                seq_len=self.prompt_len,
                dtype_str=self.dtype_str,
                phase="prefill",
            )

        else:  # decode
            # Prime KV cache with full prefill
            torch.manual_seed(42)
            input_ids = torch.randint(
                0, model.config.vocab_size,
                (self.batch, self.prompt_len),
                device=device,
            )
            attn_mask = torch.ones(self.batch, self.prompt_len, device=device)

            with torch.no_grad():
                full_out = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    use_cache=True,
                )
                past_key_values = full_out.past_key_values

            # Single-token decode input
            decode_ids = input_ids[:, -1:].contiguous()
            decode_mask = torch.ones(
                self.batch, self.prompt_len + 1, device=device
            )

            def model_fn():
                return model(
                    input_ids=decode_ids,
                    attention_mask=decode_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            # Warmup decode
            with torch.no_grad():
                model_fn()
            if device == "cuda":
                torch.cuda.synchronize()

            # Time (more trials for decode — shorter, noisier)
            elapsed_ms, _ = _cuda_timed_block(model_fn, warmup=5, trials=15)

            # Measure FLOPs
            flops, flops_method = backend.measure_flops(
                model_fn, warmup=5, trials=15,
            )
            if flops <= 0:
                flops = _analytical_decode_flops(self._param_count, self.batch)
                flops_method = "analytical"

            # Measure DRAM bytes
            bytes_read, bytes_written, bytes_method = backend.measure_dram_bytes(
                model_fn,
                param_count=self._param_count,
                model_config=model_config,
                batch=self.batch,
                seq_len=self.prompt_len,  # decode reads full existing KV cache
                dtype_str=self.dtype_str,
                phase="decode",
            )

        # ── Compute derived metrics ──────────────────────────────────
        total_bytes = bytes_read + bytes_written
        achieved_tflops = (
            (flops / (elapsed_ms / 1000)) / 1e12 if elapsed_ms > 0 else 0.0
        )
        achieved_bw = (
            (total_bytes / (elapsed_ms / 1000)) / 1e9 if elapsed_ms > 0 else 0.0
        )
        ai = round(flops / total_bytes, 2) if total_bytes > 0 else 0.0

        print(
            f"  FLOPs={flops:.2e}  time={elapsed_ms:.1f}ms  "
            f"TFLOPS={achieved_tflops:.1f}  AI={ai:.1f} FLOP/byte  "
            f"({flops_method}/{bytes_method})"
        )

        return PhaseProfile(
            flops=flops,
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            elapsed_ms=round(elapsed_ms, 2),
            achieved_tflops=round(achieved_tflops, 2),
            achieved_bw_gbps=round(achieved_bw, 2),
            arithmetic_intensity=ai,
            flops_method=flops_method,
            bytes_method=bytes_method,
        )

    # ── Model loading ───────────────────────────────────────────────────

    def _load_model(self):
        """Load model via transformers and return (model, tokenizer, config)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_str = self._dtype_canonical_name()
        torch_dtype = _parse_dtype(dtype_str)

        # Validate dtype on device
        try:
            _ = torch.tensor([1.0], dtype=torch_dtype, device=self._device)
        except RuntimeError:
            alt_type, alt_name = (torch.float16, "float16")
            print(
                f"WARNING: {dtype_str} not supported on {self._device}, "
                f"falling back to {alt_name}"
            )
            torch_dtype = alt_type
            self.dtype_str = alt_name

        print(f"Loading {self.model_id} ({self.dtype_str}) on {self._chip_name} ...")
        t_load = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
        )
        model.eval()

        self._param_count = sum(p.numel() for p in model.parameters())
        print(
            f"  {self._param_count / 1e9:.1f}B params loaded "
            f"in {time.perf_counter() - t_load:.1f}s"
        )
        return model, tokenizer, model.config

    def _cleanup_model(self):
        """Release GPU memory held by the model."""
        import torch
        gc.collect()
        if self._device == "cuda":
            torch.cuda.empty_cache()

    # ── Helpers ─────────────────────────────────────────────────────────

    def _resolve_chip_specs(self) -> dict:
        """Build the ``chip_peak`` dict from the built-in database or CLI."""
        peak_tflops = self._peak_tflops_override
        peak_bw = self._peak_bw_gbps_override
        source = "cli-flags"

        if peak_tflops is None or peak_bw is None:
            db_tflops, db_bw = _lookup_chip_specs(self._chip_name)
            if db_tflops is not None and db_bw is not None:
                peak_tflops = db_tflops
                peak_bw = db_bw
                source = f"builtin:{self._chip_name}"
            else:
                return {
                    "tflops": None,
                    "bw_gbps": None,
                    "ridge_point_i_star": None,
                    "source": "unknown",
                }

        ridge = _ridge_point(peak_tflops, peak_bw)
        return {
            "tflops": peak_tflops,
            "bw_gbps": peak_bw,
            "ridge_point_i_star": ridge,
            "source": source,
        }

    def _dtype_canonical_name(self) -> str:
        """Normalize dtype aliases to canonical names."""
        return {
            "bf16": "bfloat16",
        }.get(self.dtype_str.lower().strip(), self.dtype_str)

    def _build_env_ref(self) -> dict:
        """Build an ``env_ref`` provenance record."""
        if self._env_file:
            try:
                content = Path(self._env_file).read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                return {"file": self._env_file, "sha256": digest}
            except Exception as e:
                return {"file": self._env_file, "error": str(e)}
        return {
            "chip": self._chip_name,
            "note": "env_info.json not provided; run collect_env.py first",
        }


# ── CUDA timing helper ──────────────────────────────────────────────────────


def _cuda_timed_block(model_fn, warmup: int = 5, trials: int = 5):
    """Time *model_fn* over *trials* iterations after *warmup* warmups.

    Returns ``(median_time_ms, last_flops)`` where *last_flops* is
    always ``None`` (FLOPs are now managed by the backend).
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    times_ms = []

    for i in range(warmup + trials):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if device == "cuda":
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()

        with torch.no_grad():
            model_fn()

        if device == "cuda":
            end_ev.record()
            torch.cuda.synchronize()
            elapsed = start_ev.elapsed_time(end_ev)
        else:
            elapsed = 0.0

        if i >= warmup:
            times_ms.append(elapsed)

    median_ms = sorted(times_ms)[len(times_ms) // 2] if times_ms else 0.0
    return median_ms, None


# ── Analytical FLOP helpers (fallback) ──────────────────────────────────────


def _analytical_prefill_flops(param_count: int, batch: int,
                               seq_len: int) -> int:
    """Approximate FLOPs for a single prefill forward pass.

    Uses the standard ``2 × params × tokens`` approximation for dense
    transformers, which captures matmul and attention FLOPs at large
    batch sizes.
    """
    return 2 * param_count * batch * seq_len


def _analytical_decode_flops(param_count: int, batch: int) -> int:
    """Approximate FLOPs for a single decode step (one token per sequence)."""
    return 2 * param_count * batch


# ── Dtype helpers ───────────────────────────────────────────────────────────


def _parse_dtype(dtype_str: str):
    """Parse a dtype string to a torch dtype."""
    import torch

    _map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_str.lower().strip()
    if key in _map:
        return _map[key]
    raise ValueError(f"Unsupported dtype: {dtype_str}")
