"""
AccelMark — Moore Threads MUSA vLLM benchmark runner (vllm-musa).

Implements BenchmarkRunner for vllm-musa on Moore Threads MUSA GPUs.
See README.md in this folder for install and hardware notes.
"""

import asyncio
import gc
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
from loadgen.types import InferenceResult

import logging
logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)
logging.getLogger("vllm.engine.llm_engine").setLevel(logging.WARNING)


class MoorethreadsVLLMMUSARunner(BenchmarkRunner):
    """vLLM on Moore Threads MUSA via vllm-musa."""

    SUPPORTS_STREAMING = True
    SUPPORTS_BATCHING = True
    SUPPORTS_ONLINE = True
    SUPPORTS_MULTI_CHIP = True

    SUPPORTED_PRECISIONS = ["bf16", "fp16"]
    SUPPORTED_QUANTIZATION_BACKENDS = ["compressed-tensors"]

    _musa_runtime_prepared = False

    def __init__(self):
        self.llm = None
        self.engine = None
        self.tokenizer = None
        self.sampling_params = None
        self._loop: asyncio.AbstractEventLoop = None

    def _get_chip_count(self) -> int:
        try:
            import pymtml
            pymtml.mtmlInit()
            try:
                n = pymtml.mtmlDeviceGetCount()
            finally:
                try:
                    pymtml.mtmlShutdown()
                except Exception:
                    pass
            if n and n > 0:
                return int(n)
        except Exception:
            pass
        try:
            import torch
            n = torch.cuda.device_count()
            return n if n > 0 else 1
        except Exception:
            return 1

    def _get_framework_name(self) -> str:
        return "vllm-musa"

    def _get_framework_version(self) -> str:
        plugin_version = "unknown"
        try:
            from importlib.metadata import version
            plugin_version = version("vllm-musa")
        except Exception:
            try:
                import vllm_musa_platform  # type: ignore
                plugin_version = getattr(vllm_musa_platform, "__version__", "unknown")
            except Exception:
                pass
        try:
            import vllm
            core_version = vllm.__version__
        except Exception:
            core_version = "unknown"
        if plugin_version == "unknown" and core_version == "unknown":
            return "unknown"
        if plugin_version == "unknown":
            return core_version
        return f"{plugin_version}+vllm-{core_version}"

    def get_model_format(self) -> str:
        return "HuggingFace original"

    @classmethod
    def _prepare_musa_runtime(cls) -> None:
        if cls._musa_runtime_prepared:
            return
        import torch  # noqa: F401
        cls._musa_runtime_prepared = True

    @staticmethod
    def _legacy_vllm_musa() -> bool:
        try:
            import vllm
            ver = vllm.__version__.split("+")[0]
            major, minor = (int(x) for x in ver.split(".")[:2])
            return (major, minor) < (0, 10)
        except Exception:
            return True

    @staticmethod
    def _get_engine_arg_fields() -> set[str]:
        try:
            import dataclasses
            from vllm.engine.arg_utils import EngineArgs
            return {f.name for f in dataclasses.fields(EngineArgs)}
        except Exception:
            return set()

    def _resolve_musa_dtype(self, dtype: str, precision: str) -> str:
        if not self._legacy_vllm_musa():
            return dtype
        if dtype in ("bfloat16", "auto") or precision.upper() == "BF16":
            if dtype != "float16":
                print("  Note: vLLM 0.4.x+musa — using float16")
            return "float16"
        return dtype

    def load_model(self, model_path: str, parallelism: dict) -> None:
        self._prepare_musa_runtime()

        from transformers import AutoTokenizer
        from vllm import LLM, AsyncLLMEngine, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs

        tp_size = parallelism["tensor_parallel_size"]
        pp_size = parallelism["pipeline_parallel_size"]
        ep_size = parallelism.get("expert_parallel_size", 1)
        assert pp_size <= 1, (
            "Pipeline parallelism is not supported. Use --tensor-parallel-size."
        )

        max_tokens = parallelism["max_tokens"]
        max_model_len = parallelism["max_model_len"]
        use_async = parallelism["use_async"]
        enforce_eager = getattr(self, "_enforce_eager", False)

        cfg = getattr(self, "_runner_config", {})
        max_num_seqs = cfg.get("max_num_seqs", 256)
        musa_memory_util = cfg.get("gpu_memory_utilization", 0.85)
        extra_kwargs = dict(cfg.get("engine_kwargs") or {})

        _valid_engine_fields = self._get_engine_arg_fields()
        if _valid_engine_fields:
            _dropped = {k: v for k, v in extra_kwargs.items()
                        if k not in _valid_engine_fields}
            if _dropped:
                print(f"  Warning: engine_kwargs keys not supported by this "
                      f"vllm-musa / vLLM version and will be ignored: "
                      f"{list(_dropped)}")
            extra_kwargs = {k: v for k, v in extra_kwargs.items()
                            if k in _valid_engine_fields}

        effective_precision = getattr(self, "_effective_precision", "BF16").upper()
        precision = getattr(self, "_precision", None) or effective_precision
        _dtype_override = getattr(self, "_precision_dtype_override", None)
        _prec_eng_kwargs = dict(getattr(self, "_precision_engine_kwargs", None) or {})
        quantization = _prec_eng_kwargs.pop("quantization", None)

        _NATIVE_DTYPE_MAP = {"BF16": "bfloat16", "FP16": "float16", "FP32": "float32"}
        dtype = _NATIVE_DTYPE_MAP.get(precision, "auto")
        self._quantization_method = quantization

        if _dtype_override:
            dtype = _dtype_override
        dtype = self._resolve_musa_dtype(dtype, precision)
        if _prec_eng_kwargs:
            _prec_eng_kwargs.update(extra_kwargs)
            extra_kwargs = _prec_eng_kwargs

        if "speculative_model" in extra_kwargs and "speculative_config" not in extra_kwargs:
            extra_kwargs["speculative_config"] = {
                "model": extra_kwargs.pop("speculative_model"),
                "num_speculative_tokens": extra_kwargs.pop("num_speculative_tokens", 4),
                "draft_tensor_parallel_size": extra_kwargs.pop(
                    "speculative_draft_tensor_parallel_size", 1
                ),
            }

        print(
            f"Loading model: precision={precision}, dtype={dtype}"
            + (f", quantization_method={self._quantization_method}"
               if self._quantization_method else "")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )
        self.sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

        base_kwargs = dict(
            model=model_path,
            dtype=dtype,
            tensor_parallel_size=tp_size,
            trust_remote_code=False,
            enforce_eager=enforce_eager,
        )
        if not _valid_engine_fields or "device" in _valid_engine_fields:
            base_kwargs["device"] = "musa"
        if ep_size > 1:
            base_kwargs["enable_expert_parallel"] = True
        if quantization:
            base_kwargs["quantization"] = quantization
        if max_model_len:
            base_kwargs["max_model_len"] = max_model_len

        if not use_async:
            self.llm = LLM(**{
                **base_kwargs,
                "max_num_seqs": max_num_seqs,
                "gpu_memory_utilization": musa_memory_util,
                **extra_kwargs,
            })
        else:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            engine_args = AsyncEngineArgs(**{
                **base_kwargs,
                "gpu_memory_utilization": musa_memory_util,
                **extra_kwargs,
            })
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def get_effective_dtype(self) -> Optional[str]:
        try:
            if self.llm is not None:
                return str(self.llm.llm_engine.model_config.dtype).replace("torch.", "")
            if self.engine is not None:
                return str(self.engine.engine.model_config.dtype).replace("torch.", "")
        except Exception:
            pass
        return getattr(self, "_effective_dtype", None)

    def inference_fn_offline(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        formatted = [self._format_prompt(r.prompt) for r in requests]
        t_start = time.perf_counter()
        outputs = self.llm.generate(formatted, self.sampling_params)
        elapsed = time.perf_counter() - t_start

        self._last_accuracy_outputs = [o.outputs[0].text for o in outputs]

        return [
            InferenceResult(
                first_token_time_ms=None,
                total_time_ms=elapsed * 1000,
                output_tokens=len(o.outputs[0].token_ids),
                input_tokens=len(o.prompt_token_ids),
                success=True,
                output_text=o.outputs[0].text,
            )
            for o in outputs
        ]

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
            if first_token_time_ms is None and len(output.outputs[0].token_ids) > 0:
                first_token_time_ms = (time.perf_counter() - t_start) * 1000
            output_tokens = len(output.outputs[0].token_ids)
            output_text = output.outputs[0].text

        return InferenceResult(
            first_token_time_ms=first_token_time_ms,
            total_time_ms=(time.perf_counter() - t_start) * 1000,
            output_tokens=output_tokens,
            input_tokens=0,
            success=True,
            output_text=output_text,
        )

    async def inference_fn_token_stream(self, request: InferenceRequest):
        from vllm.utils import random_uuid

        formatted = self._format_prompt(request.prompt)
        request_id = random_uuid()
        prev_length = 0

        async for output in self.engine.generate(
            formatted, self.sampling_params, request_id
        ):
            current_text = output.outputs[0].text
            delta = current_text[prev_length:]
            if delta:
                yield delta
                prev_length = len(current_text)

    def get_peak_memory_gb(self) -> Optional[float]:
        try:
            import torch
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            pass
        try:
            import pymtml
            pymtml.mtmlInit()
            try:
                dev = pymtml.mtmlDeviceGetByIndex(0)
                info = pymtml.mtmlDeviceGetMemoryInfo(dev)
                used = getattr(info, "used", None)
                if used is not None:
                    return float(used) / (1024 ** 3)
            finally:
                try:
                    pymtml.mtmlShutdown()
                except Exception:
                    pass
        except Exception:
            pass
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
                    destroy_model_parallel,
                    destroy_distributed_environment,
                )
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass

        try:
            import torch
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

        gc.collect()

        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def parse_args(self):
        """Add vllm-musa-specific CLI flags. Base class pre-loads runner config."""
        args = super().parse_args()
        cfg = self._runner_config

        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--tensor-parallel-size", type=int, default=None,
                            dest="tensor_parallel_size")
        parser.add_argument("--expert-parallel-size", type=int, default=None,
                            dest="expert_parallel_size")
        parser.add_argument("--enforce-eager", action="store_true", default=False,
                            dest="enforce_eager")
        extra, _ = parser.parse_known_args()

        tp_size, _tp_source = self._resolve_tensor_parallel_size(
            extra.tensor_parallel_size
        )
        ep_size = (extra.expert_parallel_size
                   if extra.expert_parallel_size is not None
                   else cfg.get("expert_parallel_size", 1))
        self._enforce_eager = extra.enforce_eager or cfg.get("enforce_eager", False)

        print(f"  tensor_parallel_size = {tp_size}  [{_tp_source}]")
        if ep_size > 1:
            print(f"  expert_parallel_size = {ep_size}  [cli/yaml]")

        if not self.SUPPORTS_MULTI_CHIP and tp_size > 1:
            print(f"Warning: {self.__class__.__name__} does not support multi-chip. "
                  f"Ignoring tensor_parallel_size={tp_size}, using 1.")
            tp_size = 1
            ep_size = 1

        self._parallelism = {
            "tensor_parallel_size": tp_size,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": ep_size,
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
        if self._parallelism.get("expert_parallel_size", 1) > 1:
            extra += ["--expert-parallel-size",
                      str(self._parallelism["expert_parallel_size"])]
        if self._enforce_eager:
            extra += ["--enforce-eager"]
        return extra


if __name__ == "__main__":
    MoorethreadsVLLMMUSARunner().main()
