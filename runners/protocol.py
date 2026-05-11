"""
AccelMark Runner Protocol

Defines the minimal interface that both BenchmarkRunner (for benchmarking)
and the serve layer (for OpenAI-compatible serving) depend on.

Neither BenchmarkRunner nor serve/server.py import from each other.
Both depend only on this file and loadgen/types.py.

Any class that implements these methods satisfies RunnerProtocol structurally
(Python's runtime_checkable Protocol). No explicit inheritance required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional
from loadgen.types import InferenceResult
from runners.benchmark_runner import InferenceRequest


@runtime_checkable
class RunnerProtocol(Protocol):
    """
    Minimal interface required to run a benchmark or serve an API.

    BenchmarkRunner satisfies this protocol structurally.
    Custom runners only need to implement these methods — no import of
    BenchmarkRunner needed if you only want to use the serve layer.
    """

    # ── Capability flags ──────────────────────────────────────────────────────

    SUPPORTS_STREAMING: bool
    """True if inference_fn_streaming() is implemented and yields real tokens."""

    # ── Required methods ──────────────────────────────────────────────────────

    def load_model(self, model_path: str, suite: dict, parallelism: dict) -> None:
        """
        Load model weights into accelerator memory.

        parallelism dict always contains:
            tensor_parallel_size:   int (default 1)
            pipeline_parallel_size: int (default 1)
            expert_parallel_size:   int (default 1)
            data_parallel_size:     int (default 1)
        Read only the keys you need. Ignore unknown keys.
        """
        ...

    def release_resources(self) -> None:
        """Release accelerator memory and any distributed process groups."""
        ...

    # ── Inference ─────────────────────────────────────────────────────────────

    async def inference_fn_streaming(self, request: InferenceRequest) -> InferenceResult:
        """
        Async single-request inference. Required for serving.

        Returns a single InferenceResult after the full response is complete.
        first_token_time_ms should be set if the framework supports streaming.
        Read request.prompt at minimum — other fields are optional.
        """
        ...

    async def inference_fn_token_stream(self, request: InferenceRequest):
        """
        Async generator yielding text deltas for true SSE streaming.
        Optional — serve layer falls back to inference_fn_streaming() if
        this raises NotImplementedError.

        Yields:
            str — decoded text delta (not cumulative)
        """
        ...

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _get_framework_name(self) -> str:
        """Return the framework name, e.g. 'vLLM'."""
        ...

    def _get_framework_version(self) -> str:
        """Return the framework version string."""
        ...

    def _compute_implementation_id(self) -> Optional[str]:
        """Return the implementation ID (folder name with hash), or None."""
        ...

    def format_prompt(self, prompt: str) -> str:
        """Apply chat template or other prompt formatting."""
        ...