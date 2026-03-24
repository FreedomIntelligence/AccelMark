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
    # OFFLINE NOTE: vLLM's sync LLM.generate() is a blocking batch call — it returns
    # only after the entire batch finishes. Offline runners set this to the total batch
    # wall-clock elapsed time for every request in the batch (all requests share the
    # same value). It is NOT an individual per-request completion time. This is correct
    # for throughput computation (total_tokens / elapsed) but must not be interpreted
    # as per-request latency. See SampleRecord.total_ms for the same caveat.

    output_tokens: int
    # Actual number of tokens generated (may differ from requested max).

    success: bool
    # False if the request failed (OOM, timeout, error).
    # Failed requests are excluded from latency metrics but counted in throughput denominator.

    input_tokens: int = 0
    # Number of prompt tokens. Required for accurate offline throughput measurement
    # (total throughput = input + output tokens / elapsed, matching vLLM's metric).

    error: Optional[str] = None
    # Error message if success=False.

    output_text: Optional[str] = None
    # Generated text output. Used by _run_accuracy_integrated() for scoring.


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
    # Time to first token. None for offline (batch API gives no per-request timestamps).
    total_ms: float
    # Per-request total latency for online/interactive scenarios.
    # OFFLINE: this field holds the BATCH elapsed time (wall-clock time for the entire
    # concurrent batch, shared identically across all requests in that run). It is NOT
    # an individual completion time. Do not use offline total_ms for latency analysis;
    # use it only as a cross-check for throughput = total_tokens / (total_ms / 1000).
    success: bool
