"""
AccelMark — NVIDIA SGLang benchmark runner.

Implements BenchmarkRunner for SGLang on NVIDIA GPUs.
All orchestration logic lives in runners/benchmark_runner.py.

SGLang differences from vLLM:
  - Offline: uses sglang.Engine (sync, batched)
  - Online/interactive: uses sglang.AsyncEngine with async generator streaming
  - Streaming output is cumulative (same as vLLM) — delta sliced by prev_length
  - Quantization loaded via engine_kwargs["quantization"] or dtype="fp8"
  - Memory query uses torch.cuda.max_memory_allocated (same as vLLM)
"""

import asyncio
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



# Suppress per-request SGLang logs
import logging
logging.getLogger("sglang").setLevel(logging.WARNING)


class SGLangRunner(BenchmarkRunner):
    """AccelMark benchmark runner using SGLang on NVIDIA GPUs."""

    SUPPORTS_STREAMING  = True
    SUPPORTS_BATCHING   = True
    SUPPORTS_ONLINE     = True
    SUPPORTS_MULTI_CHIP = True

    # SGLang on NVIDIA supports all standard precisions.
    # Hardware detection in BenchmarkRunner will automatically restrict to
    # FP16 on older chips (V100, T4) that don't support BF16.
    SUPPORTED_PRECISIONS = ["bf16", "fp16", "fp32"]

    # SGLang supports the same quantization formats as vLLM via pre-quantized
    # checkpoints. FP8 requires H100 or newer (native FP8 tensor cores).
    # On A100, FP8 weights compute in BF16 — list it here; the leaderboard
    # will show effective_dtype to clarify.
    SUPPORTED_QUANTIZATION_BACKENDS = ["fp8", "compressed-tensors", "gptq_marlin"]

    def __init__(self):
        self.engine        = None   # sglang.Engine (offline/accuracy)
        self.async_engine  = None   # sglang.AsyncEngine (online/interactive)
        self.tokenizer     = None
        self._loop: asyncio.AbstractEventLoop = None
        self._sampling_params: dict = {}

    def _get_chip_count(self) -> int:
        """Return the number of available CUDA GPUs."""
        try:
            import torch
            n = torch.cuda.device_count()
            return n if n > 0 else 1
        except Exception:
            return 1

    def _get_framework_name(self) -> str:
        return "SGLang"

    def _get_framework_version(self) -> str:
        try:
            import sglang
            return sglang.__version__
        except Exception:
            return "unknown"

    def load_model(self, model_path: str, parallelism: dict) -> None:
        """Load model — sync Engine for batch inference, AsyncEngine for streaming."""
        tp_size       = parallelism["tensor_parallel_size"]
        ep_size       = parallelism.get("expert_parallel_size", 1)
        max_tokens    = parallelism["max_tokens"]
        max_model_len = parallelism["max_model_len"]
        use_async     = parallelism["use_async"]
        enforce_eager = getattr(self, "_enforce_eager", False)

        cfg                  = getattr(self, "_runner_config", {})
        mem_fraction_static  = getattr(self, "_mem_fraction_static", 0.88)
        extra_kwargs         = dict(cfg.get("engine_kwargs") or {})

        effective_precision = getattr(self, "_effective_precision", "BF16").upper()
        precision           = getattr(self, "_precision", None) or effective_precision

        _dtype_override  = getattr(self, "_precision_dtype_override", None)
        _prec_eng_kwargs = dict(getattr(self, "_precision_engine_kwargs", None) or {})
        quantization     = _prec_eng_kwargs.pop("quantization", None)

        _NATIVE_DTYPE_MAP = {"BF16": "bfloat16", "FP16": "float16", "FP32": "float32"}
        dtype = _NATIVE_DTYPE_MAP.get(precision, "auto")
        self._quantization_method = quantization

        if _dtype_override:
            dtype = _dtype_override
        if _prec_eng_kwargs:
            _prec_eng_kwargs.update(extra_kwargs)
            extra_kwargs = _prec_eng_kwargs

        print(
            f"Loading model: precision={precision}, dtype={dtype}"
            + (f", quantization_method={self._quantization_method}"
               if self._quantization_method else "")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )

        self._sampling_params = {
            "max_new_tokens": max_tokens,
            "temperature":    0.0,
        }

        engine_kwargs = dict(
            model_path=model_path,
            dtype=dtype,
            tp_size=tp_size,
            trust_remote_code=False,
            disable_cuda_graph=enforce_eager,
            mem_fraction_static=mem_fraction_static,
            **extra_kwargs,
        )
        if ep_size > 1:
            engine_kwargs["ep_size"] = ep_size
        if quantization:
            engine_kwargs["quantization"] = quantization
        if max_model_len:
            engine_kwargs["context_length"] = max_model_len

        import sglang as sgl

        if not use_async:
            self.engine = sgl.Engine(**engine_kwargs)
        else:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self.async_engine = sgl.AsyncEngine(**engine_kwargs)

    def get_effective_dtype(self) -> Optional[str]:
        """
        Report the actual compute dtype SGLang resolved after model loading.
        SGLang exposes the resolved dtype via engine.server_args.dtype.
        """
        try:
            eng = self.engine or self.async_engine
            if eng is not None:
                return str(eng.server_args.dtype).replace("torch.", "")
        except Exception:
            pass
        return getattr(self, "_effective_dtype", None)

    def inference_fn_offline(
        self, requests: list[InferenceRequest]
    ) -> list[InferenceResult]:
        """
        Synchronous batch inference via sglang.Engine.generate().

        SGLang Engine accepts a list of prompt strings and sampling params,
        returns results in the same order. total_time_ms is set to the
        wall-clock elapsed time of the entire batch — the correct denominator
        for throughput = total_tokens / elapsed.
        """
        formatted  = [self.format_prompt(r.prompt) for r in requests]
        t_start    = time.perf_counter()
        outputs    = self.engine.generate(
            prompts=formatted,
            sampling_params=self._sampling_params,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        results = []
        for req, out in zip(requests, outputs):
            # SGLang output format: {"text": str, "meta_info": {...}}
            text         = out.get("text", "") if isinstance(out, dict) else str(out)
            output_tokens = len(self.tokenizer.encode(text)) if text else 0
            input_tokens  = req.input_tokens or 0
            results.append(InferenceResult(
                first_token_time_ms=None,
                total_time_ms=elapsed_ms,
                output_tokens=output_tokens,
                input_tokens=input_tokens,
                success=True,
                output_text=text,
            ))
        return results

    async def inference_fn_streaming(
        self, request: InferenceRequest
    ) -> InferenceResult:
        """
        Async streaming inference via sglang.AsyncEngine for TTFT measurement.

        SGLang's async generator yields cumulative text (same as vLLM).
        We track prev_length and slice off deltas to count output tokens.
        first_token_time_ms is set on the first non-empty yield.
        """
        formatted           = self.format_prompt(request.prompt)
        t_start             = time.perf_counter()
        first_token_time_ms = None
        output_text         = ""
        prev_length         = 0

        async for chunk in self.async_engine.async_generate(
            prompt=formatted,
            sampling_params=self._sampling_params,
        ):
            # chunk is {"text": cumulative_text, "meta_info": {...}, "finished": bool}
            current_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            delta        = current_text[prev_length:]

            if delta and first_token_time_ms is None:
                first_token_time_ms = (time.perf_counter() - t_start) * 1000

            output_text  = current_text
            prev_length  = len(current_text)

        total_time_ms  = (time.perf_counter() - t_start) * 1000
        output_tokens  = len(self.tokenizer.encode(output_text)) if output_text else 0

        return InferenceResult(
            first_token_time_ms=first_token_time_ms,
            total_time_ms=total_time_ms,
            output_tokens=output_tokens,
            input_tokens=0,
            success=True,
            output_text=output_text,
        )

    async def inference_fn_token_stream(self, request: InferenceRequest):
        """
        Async generator yielding decoded text deltas for serve-layer SSE streaming.

        SGLang yields cumulative output — we slice off only the new delta each
        step so the serve layer receives incremental chunks, not repeated text.
        """
        formatted   = self.format_prompt(request.prompt)
        prev_length = 0

        async for chunk in self.async_engine.async_generate(
            prompt=formatted,
            sampling_params=self._sampling_params,
        ):
            current_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            delta        = current_text[prev_length:]
            if delta:
                yield delta
                prev_length = len(current_text)

    def get_peak_memory_gb(self) -> Optional[float]:
        try:
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            return None

    def release_resources(self) -> None:
        """Shut down SGLang engines and release GPU memory."""
        if self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception:
                pass
            try:
                del self.engine
            except Exception:
                pass
            self.engine = None

        if self.async_engine is not None:
            try:
                if self._loop and not self._loop.is_closed():
                    self._loop.run_until_complete(self.async_engine.shutdown())
            except Exception:
                pass
            try:
                del self.async_engine
            except Exception:
                pass
            self.async_engine = None

        # SGLang uses torch.distributed internally for tensor parallelism.
        # Destroy process group so the next engine init creates a fresh one.
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

        import gc
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def parse_args(self):
        """Add SGLang/NVIDIA-specific CLI flags. Base class pre-loads runner config."""
        args = super().parse_args()
        cfg = self._runner_config

        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--tensor-parallel-size", type=int, default=None,
                            dest="tensor_parallel_size")
        parser.add_argument("--expert-parallel-size", type=int, default=None,
                            dest="expert_parallel_size")
        parser.add_argument("--disable-cuda-graph", action="store_true", default=False,
                            dest="disable_cuda_graph")
        extra, _ = parser.parse_known_args()

        # Priority: CLI flag > yaml config > required_chips > auto-detected > default 1
        # Fully resolved by base class.
        tp_size, _tp_source = self._resolve_tensor_parallel_size(
            extra.tensor_parallel_size
        )
        ep_size = (extra.expert_parallel_size
                   if extra.expert_parallel_size is not None
                   else cfg.get("expert_parallel_size", 1))

        self._enforce_eager       = extra.disable_cuda_graph or cfg.get("disable_cuda_graph", False)
        self._mem_fraction_static = cfg.get("mem_fraction_static", 0.88)

        print(f"  tensor_parallel_size = {tp_size}  [{_tp_source}]")
        if ep_size > 1:
            print(f"  expert_parallel_size = {ep_size}  [cli/yaml]")

        self._parallelism = {
            "tensor_parallel_size":   tp_size,
            "pipeline_parallel_size": 1,
            "expert_parallel_size":   ep_size,
            "data_parallel_size":     1,
        }
        self._chip_count = tp_size
        return args

    def get_extra_subprocess_args(self, args) -> list[str]:
        """Forward SGLang/NVIDIA-specific flags to subprocess invocations."""
        extra = [
            "--tensor-parallel-size",
            str(self._parallelism.get("tensor_parallel_size", 1)),
        ]
        if self._parallelism.get("expert_parallel_size", 1) > 1:
            extra += ["--expert-parallel-size",
                      str(self._parallelism["expert_parallel_size"])]
        if self._enforce_eager:
            extra += ["--disable-cuda-graph"]
        return extra


if __name__ == "__main__":
    SGLangRunner().main()
