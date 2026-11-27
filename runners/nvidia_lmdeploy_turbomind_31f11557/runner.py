"""
AccelMark — NVIDIA LMDeploy (TurboMind) benchmark runner.

Implements BenchmarkRunner for InternLM/LMDeploy using the C++/CUDA TurboMind
backend (with automatic fallback option to PyTorch backend via runner config).
All orchestration logic lives in runners/benchmark_runner.py.
"""

import asyncio
import gc
import inspect
import sys
import time
from pathlib import Path
from typing import Optional

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import AutoTokenizer

from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
from loadgen.types import InferenceResult


class LMDeployTurboMindRunner(BenchmarkRunner):
    """AccelMark benchmark runner using LMDeploy (TurboMind) on NVIDIA GPUs."""

    SUPPORTS_STREAMING = True
    SUPPORTS_BATCHING = True
    SUPPORTS_ONLINE = True
    SUPPORTS_MULTI_CHIP = True

    SUPPORTED_PRECISIONS = ["bf16", "fp16", "fp32"]
    SUPPORTED_QUANTIZATION_BACKENDS = ["awq", "gptq"]

    def __init__(self):
        self.pipe = None
        self.tokenizer = None
        self.gen_config = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._effective_dtype: Optional[str] = None
        self._max_tokens: int = 512

    def _get_chip_count(self) -> int:
        try:
            n = torch.cuda.device_count()
            return n if n > 0 else 1
        except Exception:
            return 1

    def _get_framework_name(self) -> str:
        return "LMDeploy"

    def _get_framework_version(self) -> str:
        try:
            import lmdeploy
            return lmdeploy.__version__
        except Exception:
            return "unknown"

    def load_model(self, model_path: str, parallelism: dict) -> None:
        from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

        tp_size = parallelism["tensor_parallel_size"]
        pp_size = parallelism.get("pipeline_parallel_size", 1)
        assert pp_size <= 1, "Pipeline parallelism is not supported in LMDeployTurboMindRunner"

        max_tokens = parallelism["max_tokens"]
        max_model_len = parallelism.get("max_model_len")
        self._max_tokens = max_tokens

        cfg = getattr(self, "_runner_config", {})
        cache_max_entry_count = getattr(self, "_cache_max_entry_count", None)
        if cache_max_entry_count is None:
            cache_max_entry_count = cfg.get("cache_max_entry_count", 0.85)
        extra_kwargs = dict(cfg.get("engine_kwargs") or {})

        effective_precision = getattr(self, "_effective_precision", "BF16").upper()
        precision = getattr(self, "_precision", None) or effective_precision
        _dtype_override = getattr(self, "_precision_dtype_override", None)
        _prec_eng_kwargs = dict(getattr(self, "_precision_engine_kwargs", None) or {})
        quantization = _prec_eng_kwargs.pop("quantization", None)

        _NATIVE_DTYPE_MAP = {
            "BF16": "bfloat16",
            "FP16": "float16",
            "FP32": "float32",
        }
        dtype = _NATIVE_DTYPE_MAP.get(precision, "auto")
        if _dtype_override:
            dtype = _dtype_override
        self._effective_dtype = dtype
        self._quantization_method = quantization

        if _prec_eng_kwargs:
            _prec_eng_kwargs.update(extra_kwargs)
            extra_kwargs = _prec_eng_kwargs

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )

        # Build TurbomindEngineConfig filtering kwargs supported by installed lmdeploy
        tm_kwargs = {
            "tp": tp_size,
            "cache_max_entry_count": float(cache_max_entry_count),
            "dtype": dtype,
        }
        if max_model_len:
            tm_kwargs["session_len"] = int(max_model_len)
        tm_kwargs.update(extra_kwargs)

        try:
            sig = inspect.signature(TurbomindEngineConfig.__init__)
            valid_keys = set(sig.parameters.keys()) - {"self"}
            dropped = [k for k in tm_kwargs if k not in valid_keys]
            if dropped:
                print(f"  Warning: TurbomindEngineConfig ignoring unsupported keys: {dropped}")
                tm_kwargs = {k: v for k, v in tm_kwargs.items() if k in valid_keys}
        except Exception:
            pass

        backend_config = TurbomindEngineConfig(**tm_kwargs)
        print(
            f"Loading LMDeploy TurboMind: precision={precision}, dtype={dtype}, "
            f"tp={tp_size}, cache_max_entry_count={cache_max_entry_count}"
        )

        self.pipe = pipeline(model_path, backend_config=backend_config)
        self.gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=0.0,
            top_k=1,
            do_sample=False,
        )

    def get_effective_dtype(self) -> Optional[str]:
        return self._effective_dtype

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return max(1, len(text.split()))

    def inference_fn_offline(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        formatted = [self._format_prompt(r.prompt) for r in requests]
        t_start = time.perf_counter()
        outputs = self.pipe(formatted, gen_config=self.gen_config)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        self._last_accuracy_outputs = [getattr(o, "text", "") or "" for o in outputs]

        results = []
        for prompt_str, out in zip(formatted, outputs):
            text = getattr(out, "text", "") or ""
            out_toks = getattr(out, "generate_token_len", None)
            if out_toks is None:
                out_toks = self._count_tokens(text)
            in_toks = getattr(out, "input_token_len", None)
            if in_toks is None:
                in_toks = self._count_tokens(prompt_str)
            results.append(
                InferenceResult(
                    first_token_time_ms=None,
                    total_time_ms=elapsed_ms,
                    output_tokens=int(out_toks),
                    input_tokens=int(in_toks),
                    success=True,
                    output_text=text,
                )
            )
        return results

    async def inference_fn_streaming(self, request: InferenceRequest) -> InferenceResult:
        formatted = self._format_prompt(request.prompt)
        t_start = time.perf_counter()
        first_token_time_ms = None
        chunks: list[str] = []
        output_tokens = 0
        input_tokens = 0

        # Prefer LMDeploy's native async stream_infer if available; otherwise run in executor
        stream_gen = self.pipe.stream_infer(formatted, gen_config=self.gen_config)
        if inspect.isasyncgen(stream_gen):
            async for item in stream_gen:
                delta = getattr(item, "text", "") or ""
                if first_token_time_ms is None and delta:
                    first_token_time_ms = (time.perf_counter() - t_start) * 1000
                chunks.append(delta)
                if getattr(item, "generate_token_len", None) is not None:
                    output_tokens = int(item.generate_token_len)
                if getattr(item, "input_token_len", None) is not None:
                    input_tokens = int(item.input_token_len)
        else:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            sentinel = object()

            def _produce():
                try:
                    for item in stream_gen:
                        loop.call_soon_threadsafe(queue.put_nowait, item)
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, sentinel)

            fut = loop.run_in_executor(None, _produce)
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                delta = getattr(item, "text", "") or ""
                if first_token_time_ms is None and delta:
                    first_token_time_ms = (time.perf_counter() - t_start) * 1000
                chunks.append(delta)
                if getattr(item, "generate_token_len", None) is not None:
                    output_tokens = int(item.generate_token_len)
                if getattr(item, "input_token_len", None) is not None:
                    input_tokens = int(item.input_token_len)
            await fut

        output_text = "".join(chunks)
        if output_tokens == 0 and output_text:
            output_tokens = self._count_tokens(output_text)
        total_time_ms = (time.perf_counter() - t_start) * 1000
        return InferenceResult(
            first_token_time_ms=first_token_time_ms,
            total_time_ms=total_time_ms,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            success=True,
            output_text=output_text,
        )

    async def inference_fn_token_stream(self, request: InferenceRequest):
        formatted = self._format_prompt(request.prompt)
        stream_gen = self.pipe.stream_infer(formatted, gen_config=self.gen_config)
        if inspect.isasyncgen(stream_gen):
            async for item in stream_gen:
                delta = getattr(item, "text", "") or ""
                if delta:
                    yield delta
        else:
            for item in stream_gen:
                delta = getattr(item, "text", "") or ""
                if delta:
                    yield delta

    def get_peak_memory_gb(self) -> Optional[float]:
        try:
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            return None

    def release_resources(self) -> None:
        if self.pipe is not None:
            try:
                if hasattr(self.pipe, "close"):
                    self.pipe.close()
            except Exception:
                pass
            try:
                del self.pipe
            except Exception:
                pass
            self.pipe = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def parse_args(self):
        args = super().parse_args()
        cfg = self._runner_config
        import argparse

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--tensor-parallel-size", type=int, default=None, dest="tensor_parallel_size"
        )
        parser.add_argument(
            "--cache-max-entry-count", type=float, default=None, dest="cache_max_entry_count"
        )
        extra, _ = parser.parse_known_args()

        tp_size, _tp_source = self._resolve_tensor_parallel_size(extra.tensor_parallel_size)
        self._cache_max_entry_count = (
            extra.cache_max_entry_count
            if extra.cache_max_entry_count is not None
            else cfg.get("cache_max_entry_count", 0.85)
        )
        print(f"  tensor_parallel_size = {tp_size}  [{_tp_source}]")

        self._parallelism = {
            "tensor_parallel_size": tp_size,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "data_parallel_size": 1,
        }
        self._chip_count = tp_size
        self._precision = getattr(args, "precision", None)
        return args

    def get_extra_subprocess_args(self, args) -> list[str]:
        extra = [
            "--tensor-parallel-size",
            str(self._parallelism.get("tensor_parallel_size", 1)),
        ]
        if getattr(self, "_cache_max_entry_count", None) is not None:
            extra += ["--cache-max-entry-count", str(self._cache_max_entry_count)]
        return extra


if __name__ == "__main__":
    LMDeployTurboMindRunner().main()
