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

from typing import AsyncGenerator, AsyncIterator, Protocol, runtime_checkable, Optional
from loadgen.types import InferenceResult


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

    def load_model(self, model_path: str, suite: dict, tp_size: int) -> None:
        """Load model weights into accelerator memory."""
        ...

    def release_resources(self) -> None:
        """Release accelerator memory and any distributed process groups."""
        ...

    # ── Inference ─────────────────────────────────────────────────────────────

    async def inference_fn_streaming(self, prompt: str) -> InferenceResult:
        """
        Async single-prompt inference. Required for serving.

        Returns a single InferenceResult after the full response is complete.
        first_token_time_ms should be set if the framework supports streaming.
        """
        ...

    async def inference_fn_token_stream(
        self, prompt: str
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields decoded text tokens as they are produced.

        Optional — implement only if the framework exposes a token-level
        streaming API. When present, the serve layer uses this for true
        progressive streaming instead of sending the full response as one chunk.

        Yields:
            Decoded token strings (may be multi-character subwords, e.g. " hello")

        The serve layer falls back to inference_fn_streaming (single-chunk mode)
        if this method is not implemented.
        """
        ...
        # Make this an async generator at the type level
        # (implementations must use `yield`)
        return
        yield  # noqa: unreachable — makes type checker treat this as AsyncGenerator

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