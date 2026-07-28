"""
Huawei Ascend NPU profiling backend (stub).

The Ascend CANN ``msprof`` tool can capture operator-level FLOPs and
HBM bandwidth via the Ascend profiling infrastructure.  This stub
delegates to analytical / modelled estimates until that integration
is wired up.

See the CANN Profiling Guide for the ``msprof`` CLI and the
``profiling_op_impl.h`` API.
"""

from __future__ import annotations

from profiling.base import ProfilerBackend


class AscendProfilerBackend(ProfilerBackend):
    """Huawei Ascend NPU profiling backend.

    Currently a stub — all measurements fall through to analytical /
    modelled estimates.  Future work: integrate CANN ``msprof`` for
    operator-level FLOP counting and HBM bandwidth measurement.
    """

    ID = "ascend"
    DISPLAY_NAME = "Huawei Ascend NPU (CANN msprof)"
    PRIORITY = 30

    def is_available(self) -> bool:
        try:
            import torch_npu
            return torch_npu.npu.is_available()
        except Exception:
            return False

    def measure_flops(self, model_fn, *, warmup=5, trials=5) -> tuple[int, str]:
        # TODO: integrate CANN msprof FLOP counters
        return 0, "analytical"

    def measure_dram_bytes(
        self, model_fn, *, param_count, model_config, batch, seq_len, dtype_str, phase="prefill", **kwargs
    ) -> tuple[int, int, str]:
        # TODO: integrate CANN msprof HBM bandwidth counters
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
        try:
            import torch_npu
            props = torch_npu.npu.get_device_properties(0)
            return f"Huawei Ascend {props.name.strip()}" if props.name else "Huawei Ascend NPU"
        except Exception:
            return "Huawei Ascend NPU"
