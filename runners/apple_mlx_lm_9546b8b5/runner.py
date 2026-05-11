"""
AccelMark — Apple Silicon runner using mlx-lm (MLX).

Inference uses mlx_lm.load + stream_generate / generate, matching mlx_example.py.
Online scenario is disabled (single-device sync Metal); offline, interactive,
sustained, burst, and accuracy use streaming where applicable.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
from loadgen.types import InferenceResult


class AppleMLXLMRunner(BenchmarkRunner):
    """mlx-lm on Apple Silicon (Metal)."""

    SUPPORTS_STREAMING = True
    SUPPORTS_BATCHING = True
    SUPPORTS_ONLINE = False
    SUPPORTS_MULTI_CHIP = False

    SUPPORTED_PRECISIONS = ["bf16", "fp16", "fp32"]
    SUPPORTED_QUANTIZATION_BACKENDS: list[str] = []

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self._max_tokens: int = 512
        self._default_temp: float = 0.0

    def _get_chip_count(self) -> int:
        return 1

    def _get_framework_name(self) -> str:
        return "mlx-lm"

    def _get_framework_version(self) -> str:
        try:
            import importlib.metadata
            return importlib.metadata.version("mlx-lm")
        except Exception:
            try:
                import mlx_lm
                return str(getattr(mlx_lm, "__version__", "unknown"))
            except Exception:
                return "unknown"

    def load_model(self, model_path: str, parallelism: dict) -> None:
        from mlx_lm import load

        self._max_tokens = int(parallelism.get("max_tokens") or 512)
        self._default_temp = 0.0

        tp = parallelism.get("tensor_parallel_size", 1)
        if tp > 1:
            print(f"  Warning: mlx-lm uses a single Metal device — ignoring tensor_parallel_size={tp}")

        print(
            f"  Loading MLX model (max_tokens={self._max_tokens}, "
            f"precision={getattr(self, '_effective_precision', '?')})..."
        )

        self.model, self.tokenizer = load(
            model_path,
            tokenizer_config={"trust_remote_code": True},
        )
        self._effective_dtype = self.get_effective_dtype()

    def get_effective_dtype(self) -> Optional[str]:
        ep = getattr(self, "_effective_precision", None)
        if ep:
            return {"BF16": "bfloat16", "FP16": "float16", "FP32": "float32"}.get(
                str(ep).upper(), str(ep).lower()
            )
        return None

    def _req_max_tokens(self, request: InferenceRequest) -> int:
        if request.max_tokens is not None:
            return int(request.max_tokens)
        return self._max_tokens

    def _req_temp(self, request: InferenceRequest) -> float:
        t = request.temperature
        return float(t) if t is not None else self._default_temp

    def _run_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(
            temperature,
            top_p=1.0,
            min_p=0.0,
            min_tokens_to_keep=1,
            top_k=0,
        )
        return stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

    def _sync_streaming_inference(self, request: InferenceRequest) -> InferenceResult:
        formatted = self._format_prompt(request.prompt)
        max_t = self._req_max_tokens(request)
        temp = self._req_temp(request)

        t0 = time.perf_counter()
        first_token_time_ms: Optional[float] = None
        full_text: list[str] = []
        last_prompt_tokens = 0
        last_gen_tokens = 0

        for resp in self._run_stream(formatted, max_t, temp):
            if first_token_time_ms is None:
                first_token_time_ms = (time.perf_counter() - t0) * 1000
            full_text.append(resp.text)
            last_prompt_tokens = resp.prompt_tokens
            last_gen_tokens = resp.generation_tokens

        total_ms = (time.perf_counter() - t0) * 1000
        out = "".join(full_text)
        return InferenceResult(
            first_token_time_ms=first_token_time_ms,
            total_time_ms=total_ms,
            output_tokens=last_gen_tokens,
            input_tokens=last_prompt_tokens,
            success=True,
            output_text=out,
        )

    def inference_fn_offline(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        t_batch = time.perf_counter()
        results: list[InferenceResult] = []
        texts: list[str] = []

        for req in requests:
            formatted = self._format_prompt(req.prompt)
            max_t = self._req_max_tokens(req)
            temp = self._req_temp(req)

            parts: list[str] = []
            prompt_tokens = 0
            gen_tokens = 0
            for resp in self._run_stream(formatted, max_t, temp):
                parts.append(resp.text)
                prompt_tokens = resp.prompt_tokens
                gen_tokens = resp.generation_tokens

            text = "".join(parts)
            texts.append(text)
            ok = bool(text)
            results.append(
                InferenceResult(
                    first_token_time_ms=None,
                    total_time_ms=0.0,
                    output_tokens=gen_tokens,
                    input_tokens=prompt_tokens,
                    success=ok,
                    output_text=text if ok else None,
                    error=None if ok else "empty generation",
                )
            )

        elapsed_batch = time.perf_counter() - t_batch
        shared_ms = elapsed_batch * 1000
        for r in results:
            r.total_time_ms = shared_ms

        self._last_accuracy_outputs = texts
        return results

    async def inference_fn_streaming(self, request: InferenceRequest) -> InferenceResult:
        return await asyncio.to_thread(self._sync_streaming_inference, request)

    async def inference_fn_token_stream(self, request: InferenceRequest):
        formatted = self._format_prompt(request.prompt)
        max_t = self._req_max_tokens(request)
        temp = self._req_temp(request)
        it = iter(self._run_stream(formatted, max_t, temp))
        while True:
            try:
                resp = await asyncio.to_thread(next, it)
            except StopIteration:
                break
            if resp.text:
                yield resp.text

    def format_prompt(self, prompt: str) -> str:
        if self.tokenizer is None:
            return prompt
        chat_tmpl = getattr(self.tokenizer, "chat_template", None)
        if not chat_tmpl:
            return prompt
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    def get_peak_memory_gb(self) -> Optional[float]:
        try:
            import mlx.core as mx

            return mx.get_peak_memory() / 1e9
        except Exception:
            return None

    def release_resources(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            import gc

            import mlx.core as mx

            gc.collect()
            clear = getattr(mx, "clear_cache", None)
            if callable(clear):
                clear()
            else:
                metal = getattr(mx, "metal", None)
                if metal is not None:
                    mc = getattr(metal, "clear_cache", None)
                    if callable(mc):
                        mc()
        except Exception:
            pass

    def parse_args(self):
        args = super().parse_args()
        self._parallelism = {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "data_parallel_size": 1,
        }
        self._chip_count = 1
        return args


if __name__ == "__main__":
    AppleMLXLMRunner().main()
