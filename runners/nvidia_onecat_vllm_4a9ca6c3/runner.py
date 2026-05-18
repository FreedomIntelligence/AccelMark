"""
AccelMark — NVIDIA 1Cat-vLLM (SM70 / V100) benchmark script.

Thin vLLM runner wrapper for the 1Cat-vLLM fork on Tesla V100 / V100S.
See README.md in this folder for install, hardware scope, and tuning.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch
from vllm import LLM, AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import AutoTokenizer

from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
from loadgen.types import InferenceResult


import logging
logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)
logging.getLogger("vllm.engine.llm_engine").setLevel(logging.WARNING)


class OneCatVLLMRunner(BenchmarkRunner):
    """1Cat-vLLM on NVIDIA V100 / V100S (SM70). Use nvidia_vllm_* on newer GPUs."""

    SUPPORTS_STREAMING = True
    SUPPORTS_BATCHING = True
    SUPPORTS_ONLINE = True
    SUPPORTS_MULTI_CHIP = True

    SUPPORTED_PRECISIONS = ["fp16", "fp32"]
    SUPPORTED_QUANTIZATION_BACKENDS = ["awq"]

    def __init__(self):
        self.llm: LLM = None
        self.engine: AsyncLLMEngine = None
        self.tokenizer: AutoTokenizer = None
        self.sampling_params: SamplingParams = None
        self._loop: asyncio.AbstractEventLoop = None

    def _get_chip_count(self) -> int:
        try:
            import torch
            n = torch.cuda.device_count()
            return n if n > 0 else 1
        except Exception:
            return 1

    def _get_framework_name(self) -> str:
        return "1Cat-vLLM"

    def _get_framework_version(self) -> str:
        core = "unknown"
        try:
            import vllm
            core = vllm.__version__
        except Exception:
            pass

        fa_v100 = None
        try:
            from importlib.metadata import version as _pkg_version
            fa_v100 = _pkg_version("flash_attn_v100")
        except Exception:
            try:
                import flash_attn_v100_cuda  # type: ignore  # noqa: F401
                fa_v100 = "installed"
            except Exception:
                fa_v100 = None

        if fa_v100:
            return f"{core}+flash_attn_v100-{fa_v100}"
        return core

    def load_model(self, model_path: str, parallelism: dict) -> None:
        tp_size = parallelism["tensor_parallel_size"]
        pp_size = parallelism["pipeline_parallel_size"]
        ep_size = parallelism.get("expert_parallel_size", 1)
        assert pp_size <= 1, "Pipeline parallelism is not supported in OneCatVLLMRunner"

        max_tokens    = parallelism["max_tokens"]
        max_model_len = parallelism["max_model_len"]
        use_async     = parallelism["use_async"]
        enforce_eager = getattr(self, "_enforce_eager", False)

        cfg             = getattr(self, "_runner_config", {})
        max_num_seqs    = cfg.get("max_num_seqs", 1)
        gpu_memory_util = cfg.get("gpu_memory_utilization", 0.88)
        extra_kwargs    = dict(cfg.get("engine_kwargs") or {})

        import os
        if (
            "attention_backend" not in extra_kwargs
            and "VLLM_ATTENTION_BACKEND" not in os.environ
        ):
            extra_kwargs["attention_backend"] = "FLASH_ATTN_V100"

        try:
            import dataclasses
            from vllm.engine.arg_utils import EngineArgs as _EngineArgs
            _valid = {f.name for f in dataclasses.fields(_EngineArgs)}
            _dropped = {k: v for k, v in extra_kwargs.items() if k not in _valid}
            if _dropped:
                print(f"  Warning: engine_kwargs keys not supported by this "
                      f"1Cat-vLLM version and will be ignored: {list(_dropped)}")
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if k in _valid}
        except Exception:
            pass

        effective_precision = getattr(self, "_effective_precision", "BF16").upper()
        precision           = getattr(self, "_precision", None) or effective_precision

        _dtype_override  = getattr(self, "_precision_dtype_override", None)
        _prec_eng_kwargs = dict(getattr(self, "_precision_engine_kwargs", None) or {})

        quantization = _prec_eng_kwargs.pop("quantization", None)

        _NATIVE_DTYPE_MAP = {
            "BF16":  "bfloat16",
            "FP16":  "float16",
            "FP32":  "float32",
        }
        dtype = _NATIVE_DTYPE_MAP.get(precision, "auto")
        self._quantization_method = quantization

        if _dtype_override:
            dtype = _dtype_override

        if _prec_eng_kwargs:
            _prec_eng_kwargs.update(extra_kwargs)
            extra_kwargs = _prec_eng_kwargs

        print(f"Loading model: precision={precision}, dtype={dtype}"
              + (f", quantization_method={self._quantization_method}"
                 if self._quantization_method else ""))

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )

        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )

        if not use_async:
            llm_kwargs = dict(
                model=model_path,
                dtype=dtype,
                tensor_parallel_size=tp_size,
                trust_remote_code=False,
                enforce_eager=enforce_eager,
                max_num_seqs=max_num_seqs,
                gpu_memory_utilization=gpu_memory_util,
                **extra_kwargs,
            )
            if ep_size > 1:
                llm_kwargs["enable_expert_parallel"] = True
                llm_kwargs["tensor_parallel_size"]   = tp_size
            if quantization:
                llm_kwargs["quantization"] = quantization
            if max_model_len:
                llm_kwargs["max_model_len"] = max_model_len
            self.llm = LLM(**llm_kwargs)
        else:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            engine_kwargs = dict(
                model=model_path,
                dtype=dtype,
                tensor_parallel_size=tp_size,
                trust_remote_code=False,
                enforce_eager=enforce_eager,
                gpu_memory_utilization=gpu_memory_util,
                **extra_kwargs,
            )
            if ep_size > 1:
                engine_kwargs["enable_expert_parallel"] = True
            if max_model_len:
                engine_kwargs["max_model_len"] = max_model_len
            engine_args = AsyncEngineArgs(**engine_kwargs)
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def get_effective_dtype(self) -> Optional[str]:
        try:
            if self.llm is not None:
                dtype = self.llm.llm_engine.model_config.dtype
                return str(dtype).replace("torch.", "")
            elif self.engine is not None:
                dtype = self.engine.engine.model_config.dtype
                return str(dtype).replace("torch.", "")
        except Exception:
            pass
        return getattr(self, "_effective_dtype", None)

    def inference_fn_offline(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        formatted = [self._format_prompt(r.prompt) for r in requests]
        t_start = time.perf_counter()
        outputs = self.llm.generate(formatted, self.sampling_params)
        elapsed = time.perf_counter() - t_start

        self._last_accuracy_outputs = [o.outputs[0].text for o in outputs]

        results = []
        for output in outputs:
            results.append(InferenceResult(
                first_token_time_ms=None,
                total_time_ms=elapsed * 1000,
                output_tokens=len(output.outputs[0].token_ids),
                input_tokens=len(output.prompt_token_ids),
                success=True,
                output_text=output.outputs[0].text,
            ))
        return results

    async def inference_fn_streaming(self, request: InferenceRequest) -> InferenceResult:
        from vllm.utils import random_uuid

        formatted = self._format_prompt(request.prompt)
        request_id = random_uuid()
        t_start = time.perf_counter()
        first_token_time_ms = None
        output_tokens = 0
        output_text = ""

        async for output in self.engine.generate(
            formatted, self.sampling_params, request_id
        ):
            if (
                first_token_time_ms is None
                and len(output.outputs[0].token_ids) > 0
            ):
                first_token_time_ms = (time.perf_counter() - t_start) * 1000
            output_tokens = len(output.outputs[0].token_ids)
            output_text = output.outputs[0].text

        total_time_ms = (time.perf_counter() - t_start) * 1000
        return InferenceResult(
            first_token_time_ms=first_token_time_ms,
            total_time_ms=total_time_ms,
            output_tokens=output_tokens,
            input_tokens=0,
            success=True,
            output_text=output_text,
        )

    async def inference_fn_token_stream(self, request: InferenceRequest):
        from vllm.utils import random_uuid

        formatted   = self._format_prompt(request.prompt)
        request_id  = random_uuid()
        prev_length = 0

        async for output in self.engine.generate(
            formatted, self.sampling_params, request_id
        ):
            current_text = output.outputs[0].text
            delta = current_text[prev_length:]
            if delta:
                yield delta
                prev_length = len(current_text)

    def get_peak_memory_gb(self) -> float:
        try:
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            return None

    def release_resources(self) -> None:
        if self.llm is not None:
            try:
                del self.llm
            except Exception:
                pass
            self.llm = None

        if self.engine is not None:
            try:
                if self._loop and not self._loop.is_closed():
                    self._loop.run_until_complete(self.engine.shutdown())
            except Exception:
                pass
            try:
                del self.engine
            except Exception:
                pass
            self.engine = None

        try:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            cleanup_dist_env_and_memory(shutdown_ray=False)
        except Exception:
            try:
                from vllm.distributed.parallel_state import (
                    destroy_model_parallel, destroy_distributed_environment,
                )
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass

        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

    def parse_args(self):
        args = super().parse_args()
        cfg = self._runner_config

        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--tensor-parallel-size", type=int, default=None,
                            dest="tensor_parallel_size")
        parser.add_argument("--pipeline-parallel-size", type=int, default=None,
                            dest="pipeline_parallel_size")
        parser.add_argument("--expert-parallel-size", type=int, default=None,
                            dest="expert_parallel_size")
        parser.add_argument("--enforce-eager", action="store_true", default=False,
                            dest="enforce_eager")
        extra, _ = parser.parse_known_args()

        tp_size, _tp_source = self._resolve_tensor_parallel_size(
            extra.tensor_parallel_size
        )

        pp_size = (extra.pipeline_parallel_size
                   if extra.pipeline_parallel_size is not None
                   else cfg.get("pipeline_parallel_size", 1))
        ep_size = (extra.expert_parallel_size
                   if extra.expert_parallel_size is not None
                   else cfg.get("expert_parallel_size", 1))
        self._enforce_eager = extra.enforce_eager or cfg.get("enforce_eager", False)

        print(f"  tensor_parallel_size = {tp_size}  [{_tp_source}]")
        if ep_size > 1:
            print(f"  expert_parallel_size = {ep_size}  [cli/yaml]")

        if not self.SUPPORTS_MULTI_CHIP and tp_size * pp_size > 1:
            print(f"Warning: {self.__class__.__name__} does not support multi-chip. "
                  f"Ignoring tensor_parallel_size={tp_size}, using 1.")
            tp_size = 1
            pp_size = 1
            ep_size = 1

        self._parallelism = {
            "tensor_parallel_size":   tp_size,
            "pipeline_parallel_size": pp_size,
            "expert_parallel_size":   ep_size,
            "data_parallel_size":     1,
        }
        self._chip_count = tp_size * pp_size
        self._precision  = getattr(args, "precision", None)
        return args

    def get_extra_subprocess_args(self, args) -> list[str]:
        extra = [
            "--tensor-parallel-size",
            str(self._parallelism.get("tensor_parallel_size", 1)),
        ]
        if self._parallelism.get("pipeline_parallel_size", 1) > 1:
            extra += ["--pipeline-parallel-size",
                      str(self._parallelism["pipeline_parallel_size"])]
        if self._parallelism.get("expert_parallel_size", 1) > 1:
            extra += ["--expert-parallel-size",
                      str(self._parallelism["expert_parallel_size"])]
        if self._enforce_eager:
            extra += ["--enforce-eager"]
        return extra


if __name__ == "__main__":
    OneCatVLLMRunner().main()
