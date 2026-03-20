"""
Shared type definitions for AccelMark LoadGen.
All platform scripts must use these types.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceResult:
    """
    Result of a single inference request.
    Platform inference_fn must return a list of these.
    """
    first_token_time_ms: Optional[float]
    # Time from request submission to first output token received.
    # Must use streaming output to measure accurately.
    # Set to None if the framework does not support streaming.

    total_time_ms: float
    # Time from request submission to last output token received.

    output_tokens: int
    # Actual number of tokens generated (may differ from requested max).

    success: bool
    # False if the request failed (OOM, timeout, error).
    # Failed requests are excluded from latency metrics but counted in throughput denominator.

    error: Optional[str] = None
    # Error message if success=False.


@dataclass
class SampleRecord:
    """
    One row in samples.jsonl. Written by LoadGen automatically.
    """
    request_id: int
    batch_size: int
    scenario: str
    input_tokens: int
    output_tokens: int
    ttft_ms: Optional[float]
    total_ms: float
    success: bool
