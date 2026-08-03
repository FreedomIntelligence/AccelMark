"""
Apple Silicon (MPS / MLX) profiling backend (stub).

Apple's Metal Performance Shaders and MLX framework provide limited
public profiling APIs for FLOPs and memory bandwidth.  This stub
delegates to analytical / modelled estimates.
"""

from __future__ import annotations

from profiling.base import ProfilerBackend


class AppleProfilerBackend(ProfilerBackend):
    """Apple Silicon profiling backend.

    Currently a stub — all measurements fall through to analytical /
    modelled estimates.
    """

    ID = "apple"
    DISPLAY_NAME = "Apple Silicon (MPS)"
    PRIORITY = 40

    def is_available(self) -> bool:
        try:
            import torch
            return torch.backends.mps.is_available()
        except Exception:
            return False

    def measure_flops(self, model_fn, *, warmup=5, trials=5) -> tuple[int, str]:
        return 0, "analytical"

    def measure_dram_bytes(
        self, model_fn, *, param_count, model_config, batch, seq_len, dtype_str, phase="prefill", **kwargs
    ) -> tuple[int, int, str]:
        from profiling.backends.nvidia import (
            _compute_kv_cache_bytes,
            _compute_model_bytes,
        )
        modelled_weight = _compute_model_bytes(param_count, dtype_str)
        modelled_kv = _compute_kv_cache_bytes(model_config, batch, seq_len, dtype_str)
        total = modelled_weight + modelled_kv
        bytes_read = int(total * 0.7)
        bytes_written = total - bytes_read
        return bytes_read, bytes_written, "modelled"

    def get_chip_name(self) -> str:
        import platform
        return f"Apple {platform.processor()}"
