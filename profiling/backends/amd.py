"""
AMD GPU (ROCm) profiling backend (stub).

AMD's ROCm toolchain provides ``rocprof`` and the ROCProfiler API for
kernel-level FLOP counting and HBM bandwidth measurement.  This stub
delegates to analytical / modelled estimates until that integration
is wired up.
"""

from __future__ import annotations

from profiling.base import ProfilerBackend


class AmdProfilerBackend(ProfilerBackend):
    """AMD GPU profiling backend.

    Currently a stub — all measurements fall through to analytical /
    modelled estimates.  Future work: integrate ``rocprof`` or the
    ROCProfiler SDK for kernel-level FLOPs and memory bandwidth.
    """

    ID = "amd"
    DISPLAY_NAME = "AMD GPU (ROCm rocprof)"
    PRIORITY = 20

    def is_available(self) -> bool:
        try:
            import torch
            # PyTorch ROCm reports itself as "cuda" through the HIP
            # compatibility layer.  Check for AMD GPU in device name.
            if torch.cuda.is_available():
                name = torch.cuda.get_device_properties(0).name.lower()
                return any(t in name for t in ("amd", "radeon", "instinct", "mi"))
        except Exception:
            pass
        return False

    def measure_flops(self, model_fn, *, warmup=5, trials=5) -> tuple[int, str]:
        return 0, "analytical"

    def measure_dram_bytes(
        self, model_fn, *, param_count, model_config, batch, seq_len, dtype_str
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
