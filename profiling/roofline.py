"""
Roofline-model helpers: ridge point and bound classification.

The roofline ridge point ``I* = peak_FLOP_s / peak_byte_s`` is the
boundary between compute-bound and bandwidth-bound regimes.  These
functions are pure math — no I/O or hardware dependencies.
"""

from __future__ import annotations


def ridge_point(peak_tflops: float, peak_bw_gbps: float) -> float:
    """Compute the roofline ridge-point arithmetic intensity.

    Parameters
    ----------
    peak_tflops:
        Chip peak FP16 / BF16 TFLOPS (tensor-core).
    peak_bw_gbps:
        Chip peak HBM bandwidth in GB / s.

    Returns
    -------
    I*:
        Ridge point in FLOP / byte, rounded to two decimal places.
    """
    return round((peak_tflops * 1e12) / (peak_bw_gbps * 1e9), 2)


def classify(intensity: float, ridge: float) -> str:
    """Classify an arithmetic intensity relative to the ridge point.

    Returns ``"compute-bound"`` when *intensity* > *ridge*, otherwise
    ``"bandwidth-bound"``.
    """
    return "compute-bound" if intensity > ridge else "bandwidth-bound"
