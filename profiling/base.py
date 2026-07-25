"""
Abstract base classes and data containers for empirical arithmetic
intensity profiling.

Defines the ``ProfilerBackend`` interface that platform-specific
backends must implement, and the structured dataclasses returned by
the profiling pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ── Data containers ──────────────────────────────────────────────────────────


@dataclass
class PhaseProfile:
    """Profile data for a single inference phase (prefill or decode).

    Attributes
    ----------
    flops:
        Total floating-point operations for one forward pass.
    bytes_read:
        Bytes read from DRAM / HBM during the forward pass.
    bytes_written:
        Bytes written to DRAM / HBM during the forward pass.
    elapsed_ms:
        Median GPU wall-clock time in milliseconds (CUDA-event fenced).
    achieved_tflops:
        Effective TFLOPS = flops / elapsed.
    achieved_bw_gbps:
        Effective bandwidth = (bytes_read + bytes_written) / elapsed.
    arithmetic_intensity:
        FLOP / (bytes_read + bytes_written).
    flops_method:
        Provenance tag: ``"measured"`` (FlopCounterMode or equivalent),
        ``"analytical"`` (``2 × params × tokens``).
    bytes_method:
        Provenance tag: ``"profiler"`` (PyTorch CUPTI profiler),
        ``"profiler+modelled"`` (profiler base with modelled KV cache),
        ``"modelled"`` (weight footprint + KV cache estimate).
    memory_usage_bytes:
        Peak device memory allocation during the phase (optional).
    """

    flops: int
    bytes_read: int = 0
    bytes_written: int = 0
    elapsed_ms: float = 0.0
    achieved_tflops: float = 0.0
    achieved_bw_gbps: float = 0.0
    arithmetic_intensity: float = 0.0
    flops_method: str = "analytical"
    bytes_method: str = "analytical"
    memory_usage_bytes: Optional[int] = None
    time_frac: float = 0.0

    @property
    def total_bytes(self) -> int:
        """Total DRAM traffic = bytes_read + bytes_written."""
        return self.bytes_read + self.bytes_written


@dataclass
class IntensityResult:
    """Complete intensity profiling result for a (chip, suite) pair.

    This is the top-level output of :class:`IntensityProfiler.run`.
    """

    suite: str
    chip: str
    model: str
    operating_point: dict = field(default_factory=dict)
    prefill: Optional[PhaseProfile] = None
    decode: Optional[PhaseProfile] = None
    blended: Optional[dict] = None
    chip_peak: Optional[dict] = None
    classification: dict = field(default_factory=dict)
    env_ref: dict = field(default_factory=dict)
    runtime_note: str = ""

    def to_dict(self) -> dict:
        """Serialize to the JSON-compatible dict expected by the CLI."""

        def _phase_dict(p: Optional[PhaseProfile]) -> dict:
            if p is None:
                return {}
            return {
                "flops": p.flops,
                "bytes": p.total_bytes,
                "bytes_read": p.bytes_read,
                "bytes_written": p.bytes_written,
                "arithmetic_intensity": p.arithmetic_intensity,
                "achieved_tflops": p.achieved_tflops,
                "achieved_bw_gbps": p.achieved_bw_gbps,
                "time_ms_median": p.elapsed_ms,
                "time_frac": p.time_frac,
                "flops_method": p.flops_method,
                "bytes_method": p.bytes_method,
            }

        return {
            "suite": self.suite,
            "chip": self.chip,
            "model": self.model,
            "operating_point": self.operating_point,
            "phases": {
                "prefill": _phase_dict(self.prefill),
                "decode": _phase_dict(self.decode),
            },
            "blended": self.blended or {},
            "chip_peak": self.chip_peak or {},
            "classification": self.classification,
            "env_ref": self.env_ref,
            "runtime_note": self.runtime_note,
        }


# ── Abstract backend ────────────────────────────────────────────────────────


class ProfilerBackend(ABC):
    """Platform-specific measurement backend.

    Each backend provides hardware-aware FLOP counting and DRAM-byte
    measurement for its accelerator family.  Backends are discovered
    and registered via the ``profiling.backends`` auto-discovery
    mechanism (mirrors ``runners/platforms/``).

    Subclasses **must** set class-level attributes:

    ``ID``
        Short lowercase identifier, e.g. ``"nvidia"``.
    ``DISPLAY_NAME``
        Human-readable label for diagnostics and CLI output.
    ``PRIORITY``
        Integer detection order.  Lower = tried first.  Default 50.

    Lifecycle
    ---------
    The orchestrator calls :meth:`setup` before profiling and
    :meth:`teardown` after, so backends that install monkey-patches
    or profiler hooks can acquire / release cleanly.
    """

    ID: str = ""
    DISPLAY_NAME: str = ""
    PRIORITY: int = 50

    # ── Detection ────────────────────────────────────────────────────────

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if this backend's hardware / runtime is usable.

        Typical implementations check ``torch.cuda.is_available()``,
        ``torch_npu.npu.is_available()``, etc.
        """
        ...

    # ── Lifecycle ────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Called once before profiling begins.

        Use for one-time initialization such as monkey-patching
        PyTorch FLOP counters or configuring profiler hooks.
        Default is a no-op.
        """

    def teardown(self) -> None:
        """Called once after profiling completes.

        Use to restore monkey-patches or clean up profiler state.
        Default is a no-op.
        """

    # ── Measurement ──────────────────────────────────────────────────────

    @abstractmethod
    def measure_flops(
        self,
        model_fn,
        *,
        warmup: int = 5,
        trials: int = 5,
    ) -> tuple[int, str]:
        """Measure FLOPs for one forward pass of *model_fn*.

        Parameters
        ----------
        model_fn:
            Zero-argument callable that runs one forward pass.
        warmup:
            Number of warm-up iterations (not counted).
        trials:
            Number of measurement iterations.

        Returns
        -------
        (flops, method):
            *flops* is the total floating-point operations (median
            across trials).  *method* is a provenance tag —
            ``"measured"`` or ``"analytical"``.
        """
        ...

    @abstractmethod
    def measure_dram_bytes(
        self,
        model_fn,
        *,
        param_count: int,
        model_config,
        batch: int,
        seq_len: int,
        dtype_str: str,
    ) -> tuple[int, int, str]:
        """Measure DRAM traffic for one forward pass of *model_fn*.

        Returns
        -------
        (bytes_read, bytes_written, method):
            *bytes_read* and *bytes_written* are the measured or
            modelled DRAM traffic.  *method* is a provenance tag —
            ``"profiler"``, ``"profiler+modelled"``, or ``"modelled"``.
        """
        ...

    # ── Chip specs ───────────────────────────────────────────────────────

    def get_chip_name(self) -> str:
        """Return the detected chip name string.

        The default implementation queries PyTorch CUDA properties.
        Override for non-CUDA platforms.
        """
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
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return out.strip().splitlines()[0].strip()
        except Exception:
            pass
        return "unknown"


class BackendNotAvailableError(RuntimeError):
    """Raised when no profiling backend is available for the current platform."""
