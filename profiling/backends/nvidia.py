"""
NVIDIA GPU profiling backend.

Provides hardware-measured FLOP counting (via PyTorch
``FlopCounterMode`` with a GQA monkey-patch and causal-mask correction)
and analytical DRAM-byte estimation from model geometry.

FLOPs are measured; bytes are computed analytically because
``torch.profiler``'s ``self_device_memory_usage`` measures GPU
allocation volume, not actual DRAM traffic.
"""

from __future__ import annotations

import gc
from typing import Optional

from profiling.base import ProfilerBackend


class NvidiaProfilerBackend(ProfilerBackend):
    """NVIDIA GPU profiling backend.

    Supports FLOP measurement via ``FlopCounterMode`` (patched for
    Grouped-Query Attention and causal-mask corrected) and analytical
    DRAM-byte estimation from model geometry.

    Gracefully degrades to analytical FLOPs when hardware counters
    are unavailable.
    """

    ID = "nvidia"
    DISPLAY_NAME = "NVIDIA GPU (CUDA)"
    PRIORITY = 10

    def __init__(self):
        self._original_sdpa: Optional[callable] = None
        self._original_sdpa_bwd: Optional[callable] = None

    # ── Detection ────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def setup(self) -> None:
        _patch_sdpa_flop_count_for_gqa()

    def teardown(self) -> None:
        _restore_sdpa_flop_count()

    # ── FLOP measurement ─────────────────────────────────────────────────

    def measure_flops(
        self,
        model_fn,
        *,
        warmup: int = 5,
        trials: int = 5,
    ) -> tuple[int, str]:
        """Measure FLOPs for one forward pass.

        Uses ``FlopCounterMode`` (patched for GQA with causal-mask
        correction).  Falls back to analytical if the counter is
        unavailable.
        """
        import torch

        FlopCounterMode = _get_flop_counter_mode()
        if FlopCounterMode is None:
            return 0, "analytical"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        flop_ok = True
        measured_flops_values: list[int] = []

        for i in range(warmup + trials):
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            use_flops = flop_ok
            flop_ctx = FlopCounterMode(display=False) if use_flops else None

            try:
                with (flop_ctx if flop_ctx else _NullContext()):
                    with torch.no_grad():
                        model_fn()
                if device == "cuda":
                    torch.cuda.synchronize()
            except AssertionError:
                flop_ok = False
                flop_ctx = None
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                with torch.no_grad():
                    model_fn()
                if device == "cuda":
                    torch.cuda.synchronize()

            if i >= warmup and flop_ctx is not None and flop_ok:
                try:
                    f = flop_ctx.get_total_flops()
                    if f and f > 0:
                        measured_flops_values.append(f)
                except Exception:
                    pass

        if measured_flops_values:
            median_flops = sorted(measured_flops_values)[len(measured_flops_values) // 2]
            return median_flops, "measured"

        return 0, "analytical"

    # ── DRAM byte measurement ────────────────────────────────────────────

    def measure_dram_bytes(
        self,
        model_fn,
        *,
        param_count: int,
        model_config,
        batch: int,
        seq_len: int,
        dtype_str: str,
        phase: str = "prefill",
    ) -> tuple[int, int, str]:
        """Estimate DRAM traffic analytically from model geometry.

        Counts three components:
        1. **Weight reads** — full model weights, read once per forward pass.
        2. **KV-cache traffic** — reads/writes to the per-layer key-value store.
        3. **Activation traffic** — input/output tensors of attention and MLP
           projections (FlashAttention tiling assumed: Q/K/V intermediates
           and attention scores stay in SRAM, not HBM).

        Intermediate activation tensors are included here because they scale
        predictably with model geometry, unlike the previous model which
        excluded them as "implementation-dependent".  The per-layer formula
        follows the FlashAttention HBM traffic model: O(B·S·(4d + 2·d_ff)).

        **Prefill**: reads weights + writes new KV + full activation traffic.
        **Decode**:  reads weights + existing KV + single-token activation.

        Returns ``(bytes_read, bytes_written, "analytical")``.
        """
        weight_bytes = _compute_weight_bytes(param_count, dtype_str)
        kv_bytes_per_token = _kv_bytes_per_token(model_config, dtype_str)
        act_bytes = _compute_activation_bytes(
            model_config, batch, seq_len, dtype_str, phase=phase
        )

        if phase == "decode":
            # Decode: reads weights + full existing KV cache,
            #         writes 1 new token's KV.
            kv_read = kv_bytes_per_token * seq_len * batch
            kv_write = kv_bytes_per_token * batch
            bytes_read = weight_bytes + kv_read + act_bytes
            bytes_written = kv_write
        else:
            # Prefill: reads weights (amortised over batch),
            #          writes full KV cache for all tokens.
            bytes_read = weight_bytes + act_bytes
            bytes_written = kv_bytes_per_token * seq_len * batch

        return bytes_read, bytes_written, "analytical"


# ── GQA + causal-mask monkey-patch for FlopCounterMode ────────────────────────

_ORIGINAL_SDPA: Optional[callable] = None
_ORIGINAL_SDPA_BWD: Optional[callable] = None


def _patch_sdpa_flop_count_for_gqa() -> None:
    """Monkey-patch PyTorch's ``sdpa_flop_count`` to support GQA and
    correct for causal masking.

    **GQA fix**: The upstream function asserts ``h == _h2 == _h3``. Under
    GQA, ``query_heads`` is a multiple of ``kv_heads`` (e.g. Llama-3-8B
    has 32 query heads but only 8 KV heads).  We relax this to only
    require ``_h2 == _h3``.  The existing ``bmm_flop`` calls already use
    ``b * h`` from the query shape, which is correct.

    **Causal-mask correction**: ``FlopCounterMode`` counts full ``S x S``
    attention pairs, but causal self-attention only computes the lower
    triangle — ``S(S+1)/2`` pairs.  We apply a scaling factor of
    ``(s_q + 1) / (2 * s_k)`` when the shapes indicate self-attention
    (``s_q == s_k``, which implies causal masking for decoder-only
    models).
    """
    global _ORIGINAL_SDPA, _ORIGINAL_SDPA_BWD

    try:
        import torch.utils.flop_counter as fc_mod
    except ImportError:
        return

    if _ORIGINAL_SDPA is not None:
        return

    _ORIGINAL_SDPA = fc_mod.sdpa_flop_count

    def _patched_sdpa_flop_count(query_shape, key_shape, value_shape):
        """GQA-aware + causal-corrected sdpa FLOP count."""
        b, h, s_q, d_q = query_shape
        _b2, _h2, s_k, _d2 = key_shape
        _b3, _h3, _s3, d_v = value_shape
        # GQA fix: only KV/value heads must match each other
        assert b == _b2 == _b3 and _h2 == _h3 and d_q == _d2 and s_k == _s3
        total_flops = 0
        # Q @ K^T
        total_flops += fc_mod.bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
        # scores @ V
        total_flops += fc_mod.bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_v))

        # Causal-mask correction: FlopCounterMode counts full SxS, but
        # causal self-attention only computes S(S+1)/2 pairs.  Apply
        # when it's self-attention (s_q == s_k — all decoder-only models).
        if s_q == s_k and s_q > 1:
            causal_scale = (s_q + 1) / (2 * s_q)
            total_flops = int(total_flops * causal_scale)

        return total_flops

    fc_mod.sdpa_flop_count = _patched_sdpa_flop_count

    # Also patch sdpa_backward_flop_count (line 444) — same GQA + causal fix
    if hasattr(fc_mod, "sdpa_backward_flop_count"):
        _ORIGINAL_SDPA_BWD = fc_mod.sdpa_backward_flop_count

        def _patched_sdpa_backward_flop_count(grad_out_shape, query_shape,
                                               key_shape, value_shape):
            b, h, s_q, d_q = query_shape
            _b2, _h2, s_k, _d2 = key_shape
            _b3, _h3, _s3, d_v = value_shape
            _b4, _h4, _s4, _d4 = grad_out_shape
            assert (b == _b2 == _b3 == _b4
                    and _h2 == _h3 == _h4
                    and d_q == _d2
                    and d_v == _d4
                    and s_k == _s3
                    and s_q == _s4)
            total_flops = 0
            total_flops += fc_mod.bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
            total_flops += fc_mod.bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_v))
            total_flops += fc_mod.bmm_flop((b * h, s_q, d_v), (b * h, d_v, s_k))
            total_flops += fc_mod.bmm_flop((b * h, s_q, s_k), (b * h, d_q, s_k))
            if s_q == s_k and s_q > 1:
                causal_scale = (s_q + 1) / (2 * s_q)
                total_flops = int(total_flops * causal_scale)
            return total_flops

        fc_mod.sdpa_backward_flop_count = _patched_sdpa_backward_flop_count


def _restore_sdpa_flop_count() -> None:
    """Restore the original ``sdpa_flop_count`` functions."""
    global _ORIGINAL_SDPA, _ORIGINAL_SDPA_BWD
    try:
        import torch.utils.flop_counter as fc_mod
    except ImportError:
        return
    if _ORIGINAL_SDPA is not None:
        fc_mod.sdpa_flop_count = _ORIGINAL_SDPA
        _ORIGINAL_SDPA = None
    if _ORIGINAL_SDPA_BWD is not None:
        fc_mod.sdpa_backward_flop_count = _ORIGINAL_SDPA_BWD
        _ORIGINAL_SDPA_BWD = None


# ── Analytical DRAM-byte helpers ──────────────────────────────────────────────


def _compute_weight_bytes(param_count: int, dtype_str: str) -> int:
    """Model weight footprint in bytes (one full read per forward pass)."""
    bytes_per_param = 1 if "int8" in dtype_str else (4 if "float32" in dtype_str else 2)
    return param_count * bytes_per_param


def _compute_activation_bytes(
    model_config, batch: int, seq_len: int, dtype_str: str, phase: str = "prefill"
) -> int:
    """Analytical activation HBM traffic for one forward pass.

    Counts the input/output tensor traffic through attention and MLP
    projections.  Assumes FlashAttention tiling (Q/K/V intermediates and
    attention scores stay in SRAM, not HBM).

    Per-layer HBM traffic (prefill):
      - Read:  layer input (for QKV + gate/up projections, reused in L2)
      - Write: attention output (d), MLP intermediate (d_ff), layer output (d)
      - Read:  MLP intermediate (for down projection)
      Total ≈ B × S × (4d + 2×d_ff) × dtype_size

    Per-layer HBM traffic (decode, single new token):
      Total ≈ B × (4d + 2×d_ff) × dtype_size
    """
    d = getattr(model_config, "hidden_size", 0)
    d_ff = getattr(model_config, "intermediate_size", 0)
    num_layers = getattr(model_config, "num_hidden_layers", 0)
    bytes_per_elem = 1 if "int8" in dtype_str else (4 if "float32" in dtype_str else 2)

    if phase == "decode":
        # Single new token per sequence: S=1
        per_layer = batch * (4 * d + 2 * d_ff)
    else:
        per_layer = batch * seq_len * (4 * d + 2 * d_ff)

    return per_layer * num_layers * bytes_per_elem


def _kv_bytes_per_token(model_config, dtype_str: str) -> int:
    """KV-cache bytes per token (Key + Value for all layers).

    Formula: 2 * n_layers * n_kv_heads * head_dim * dtype_size

    For Llama-3-8B: 2 * 32 * 8 * 128 * 2 = 131072 bytes = 128 KiB/token.
    """
    num_layers = getattr(model_config, "num_hidden_layers", 0)
    num_kv_heads = getattr(
        model_config,
        "num_key_value_heads",
        getattr(model_config, "num_attention_heads", 0),
    )
    head_dim = (
        getattr(model_config, "hidden_size", 0)
        // max(getattr(model_config, "num_attention_heads", 1), 1)
    )
    bytes_per_elem = 1 if "int8" in dtype_str else (4 if "float32" in dtype_str else 2)
    return 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem


# ── Internal helpers ─────────────────────────────────────────────────────────


def _get_flop_counter_mode():
    """Return ``FlopCounterMode`` or ``None`` if unavailable."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        return FlopCounterMode
    except ImportError:
        return None


class _NullContext:
    """No-op context manager — fallback when FlopCounterMode is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
