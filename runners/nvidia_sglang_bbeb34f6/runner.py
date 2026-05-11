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
import os
# 允许 SGLang 在推测解码时覆盖派生的上下文长度限制
os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
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
        # 合并 precision engine kwargs
        if _prec_eng_kwargs:
            _prec_eng_kwargs.update(extra_kwargs)
            extra_kwargs = _prec_eng_kwargs

        
        #新增
        # ===== 处理 speculative 场景参数（将 AccelMark 参数转换为 SGLang 参数）=====
        # AccelMark 的 benchmark_runner 会传入 speculative_model 参数
        # 但 SGLang 需要的是 speculative_draft_model_path 和 speculative_algorithm

        # 1. 提取并移除 speculative_model（这是 draft model 路径）
        speculative_model_path = extra_kwargs.pop("speculative_model", None)

        # 2. 提取 num_speculative_tokens 作为参考值（但不直接传给 SGLang）
        num_speculative_tokens = extra_kwargs.pop("num_speculative_tokens", None)

        # 3. 移除 SGLang 不支持的参数
        extra_kwargs.pop("speculative_draft_tensor_parallel_size", None)

        # 4. 如果有 draft model，设置 SGLang 所需的参数
        if speculative_model_path:
            # SGLang 要求明确指定推测解码算法[citation:2]
            extra_kwargs["speculative_algorithm"] = "EAGLE"
            extra_kwargs["speculative_draft_model_path"] = speculative_model_path

            mem_fraction_static = 0.7
            print(f"Speculative decoding: reducing mem_fraction_static to {mem_fraction_static}")

            # 可选：将 AccelMark 的 num_speculative_tokens 转换为 SGLang 参数
            # SGLang 默认会自动调优，如果需要显式指定可以取消注释
            #if num_speculative_tokens:
            #    extra_kwargs["speculative_num_draft_tokens"] = num_speculative_tokens
            
            print(f"Speculative decoding enabled: algorithm=EAGLE, draft_model={speculative_model_path}")
            #if num_speculative_tokens:
            #    print(f"  speculative_num_draft_tokens = {num_speculative_tokens}")

        # 调试：打印转换后的参数
        print("DEBUG: extra_kwargs after conversion =", list(extra_kwargs.keys()))
        # 注意：不需要过滤 speculative_draft_model_path 等参数
        # 因为上面已经将 speculative_model 转换为了 SGLang 能识别的参数
        # ============================================

 

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
        #================================
        print("DEBUG: extra_kwargs keys =", list(extra_kwargs.keys()))
        #====================================
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
        from sglang.srt.entrypoints.engine import Engine
        #if not use_async:
        #    self.engine = sgl.Engine(**engine_kwargs)
        #else:
        #    self._loop = asyncio.new_event_loop()
        #    asyncio.set_event_loop(self._loop)
        #    #self.async_engine = sgl.AsyncEngine(**engine_kwargs)
        #    #新增
        #    self.engine=Engine(**engine_kwargs)
        
        #新增
        #================
        self.engine = sgl.Engine(**engine_kwargs)
        if use_async:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            #self.async_engine = sgl.AsyncEngine(**engine_kwargs)
            #新增
        #=============================
            

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
            prompt=formatted,
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
        #新增
        #======================
        # 确保 sampling_params 中有 stream=True
        sampling_params = self._sampling_params.copy()
        #sampling_params["stream"] = True
        # ==================== 新增修复逻辑 ====================
        # 计算剩余可用 token 空间 (30208 - 29968 = 240)
        # 为了保险起见，我们将最大生成 token 数压低至 224
        if sampling_params.get("max_new_tokens", 256) > 224:
            sampling_params["max_new_tokens"] = 224
        # ====================================================
        # 2. 检查并移除可能存在的 "stream" 键，防止 SamplingParams 初始化报错
        if "stream" in sampling_params:
            del sampling_params["stream"]
        # 修复：先 await 获取异步生成器
        generator = await self.engine.async_generate(
            prompt=formatted,
            sampling_params=sampling_params,
            #sampling_params=self._sampling_params,
            stream=True,  # 关键：stream 是 async_generate 的参数
        )
        #=======================
        #async for chunk in self.async_engine.async_generate(
        #新增
        if hasattr(generator, '__aiter__'):
            async for chunk in generator:
        #====================
                # chunk is {"text": cumulative_text, "meta_info": {...}, "finished": bool}
                current_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                delta        = current_text[prev_length:]

                if delta and first_token_time_ms is None:
                    first_token_time_ms = (time.perf_counter() - t_start) * 1000

                output_text  = current_text
                prev_length  = len(current_text)
        else:
            # 如果是直接返回的完整结果（非流式）
            current_text = generator.get("text", "") if isinstance(generator, dict) else str(result)
            output_text = current_text
            first_token_time_ms = (time.perf_counter() - t_start) * 1000

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

        #新增 同样需要先 await
        #=================================
        generator = await self.engine.async_generate(
            prompt=formatted,
            sampling_params=self._sampling_params,
            stream=True,
        )

        async for chunk in generator:
        #==========================================
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
