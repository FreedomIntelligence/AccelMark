"""
Mock runner for testing the serve layer without a GPU or real model.

Satisfies RunnerProtocol structurally — no import of BenchmarkRunner needed.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from loadgen.types import InferenceResult
from runners.benchmark_runner import InferenceRequest


class MockRunner:
    """
    Minimal RunnerProtocol-compatible runner for testing.
    Returns configurable fake responses with no GPU required.
    """

    SUPPORTS_STREAMING  = True
    SUPPORTS_BATCHING   = True
    SUPPORTS_ONLINE     = True
    SUPPORTS_MULTI_CHIP = False

    def __init__(
        self,
        response_text: str = "Hello from mock runner.",
        output_tokens: int = 8,
        input_tokens: int = 12,
        ttft_ms: float = 42.0,
        latency_ms: float = 120.0,
        impl_id: str = "nvidia_mock_deadbeef",
    ):
        self._response_text = response_text
        self._output_tokens = output_tokens
        self._input_tokens  = input_tokens
        self._ttft_ms       = ttft_ms
        self._latency_ms    = latency_ms
        self._impl_id       = impl_id
        self._loaded        = False

    def load_model(self, model_path: str, suite: dict, parallelism: dict) -> None:
        """parallelism dict contains tensor_parallel_size, pipeline_parallel_size, etc."""
        self._loaded = True

    def release_resources(self) -> None:
        self._loaded = False

    def inference_fn_offline(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        return [
            InferenceResult(
                first_token_time_ms=self._ttft_ms,
                total_time_ms=self._latency_ms,
                output_tokens=self._output_tokens,
                input_tokens=self._input_tokens,
                success=True,
                output_text=self._response_text,
            )
            for _ in requests
        ]

    async def inference_fn_streaming(self, request: InferenceRequest) -> InferenceResult:
        """Read request.prompt if needed. Simulates a small async delay."""
        await asyncio.sleep(0.001)
        return InferenceResult(
            first_token_time_ms=self._ttft_ms,
            total_time_ms=self._latency_ms,
            output_tokens=self._output_tokens,
            input_tokens=self._input_tokens,
            success=True,
            output_text=self._response_text,
        )

    async def inference_fn_token_stream(self, request: InferenceRequest):
        """Yield response word by word to simulate token streaming."""
        for word in self._response_text.split():
            await asyncio.sleep(0.001)
            yield word + " "

    def format_prompt(self, prompt: str) -> str:
        return prompt  # pass through unchanged

    def _get_framework_name(self) -> str:
        return "MockFramework"

    def _get_framework_version(self) -> str:
        return "0.0.1"

    def _compute_implementation_id(self) -> Optional[str]:
        return self._impl_id


class NoStreamingMockRunner(MockRunner):
    """Mock runner that declares SUPPORTS_STREAMING = False."""
    SUPPORTS_STREAMING = False