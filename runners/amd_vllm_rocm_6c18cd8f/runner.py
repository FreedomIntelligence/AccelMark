"""
AccelMark — AMD ROCm vLLM benchmark runner.

Implements BenchmarkRunner for vLLM on AMD GPUs via ROCm.
All orchestration logic lives in runners/benchmark_runner.py.

ROCm vs NVIDIA vLLM differences:
  - Install: use vllm[rocm] or the ROCm-specific wheel from vllm-project/vllm
  - torch.cuda.* APIs work on ROCm (HIP aliased as CUDA) — no API changes needed
  - FP8 is supported on MI300X only (gfx942). Excluded by default in
    SUPPORTED_QUANTIZATIONS for safety; MI300X users can override by
    subclassing and setting SUPPORTED_QUANTIZATIONS = ["fp8", "w8a8", "w8a16", "w4a16"]
  - BF16 is supported on MI200 series and newer (gfx90a+)
  - MI100 (gfx908) does NOT support BF16 — hardware detection handles fallback to FP16
  - Tensor parallelism via RCCL (ROCm equivalent of NCCL) — same API as NCCL

Installation:
    # Install ROCm toolkit first: https://rocm.docs.amd.com/
    # Then install vLLM ROCm wheel:
    pip install vllm[rocm]
    # or follow: https://docs.vllm.ai/en/latest/getting_started/amd-installation.html

    pip install -r runners/amd_vllm_rocm_{hash8}/requirements.txt

Usage:
    python run.py --runner amd_vllm_rocm_{hash8} --suite suite_A
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
from vllm import LLM, AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import AutoTokenizer

from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
from loadgen.types import InferenceResult



# Suppress per-request vLLM logs
import logging
logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)
logging.getLogger("vllm.engine.llm_engine").setLevel(logging.WARNING)


class AMDVLLMROCmRunner(BenchmarkRunner):
    """
    AccelMark benchmark runner using vLLM on AMD GPUs via ROCm.

    The implementation is almost identical to the NVIDIA vLLM runner.
    The key behavioral differences are in capability flags and quantization
    support — all inference code is unchanged because vLLM's ROCm backend
    aliases torch.cuda.* to HIP and keeps the same Python API.
    """

    SUPPORTS_STREAMING  = True
    SUPPORTS_BATCHING   = True
    SUPPORTS_ONLINE     = True
    SUPPORTS_MULTI_CHIP = True   # RCCL-based tensor parallelism

    # BF16 supported on MI200 series and newer (gfx90a+).
    # MI100 (gfx908) is FP16-only — BenchmarkRunner's hardware detection
    # handles the fallback automatically via the chip name lookup.
    SUPPORTED_PRECISIONS = ["bf16", "fp16", "fp32"]

    # FP8 is excluded by default for broad MI-series compatibility.
    # MI300X (gfx942) does support native FP8 — users on MI300X can enable it
    # by subclassing and overriding:
    #
    #   class MI300XVLLMROCmRunner(AMDVLLMROCmRunner):
    #       SUPPORTED_QUANTIZATION_BACKENDS = ["fp8", "compressed-tensors", "gptq_marlin"]
    #
    # compressed-tensors (W8A8/W8A16) and gptq_marlin (W4A16) confirmed on MI250X/MI300X.
    SUPPORTED_QUANTIZATION_BACKENDS = ["compressed-tensors", "gptq_marlin"]

    def __init__(self):
        self.llm:             LLM               = None
        self.engine:          AsyncLLMEngine     = None
        self.tokenizer:       AutoTokenizer      = None
        self.sampling_params: SamplingParams     = None
        self._loop:           asyncio.AbstractEventLoop = None

    def _get_chip_count(self) -> int:
        """Return the number of available ROCm/CUDA GPUs."""
        try:
            import torch
            n = torch.cuda.device_count()
            return n if n > 0 else 1
        except Exception:
            return 1

    def _get_framework_name(self) -> str:
        return "vLLM-ROCm"

    def _get_framework_version(self) -> str:
        try:
            import vllm
            return vllm.__version__
        except Exception:
            return "unknown"

    def load_model(self, model_path: str, parallelism: dict) -> None:
        """
        Load model onto AMD GPU via vLLM ROCm backend.

        The constructor kwargs are identical to the NVIDIA vLLM runner.
        vLLM's ROCm backend activates automatically when HIP/ROCm is the
        runtime — no explicit device flag is needed.
        """
        tp_size       = parallelism["tensor_parallel_size"]
        pp_size       = parallelism["pipeline_parallel_size"]
        ep_size       = parallelism.get("expert_parallel_size", 1)
        # vLLM ROCm does not support pipeline parallelism (same limitation as CUDA).
        assert pp_size <= 1, "Pipeline parallelism is not supported in AMDVLLMROCmRunner"

        max_tokens    = parallelism["max_tokens"]
        max_model_len = parallelism["max_model_len"]
        use_async     = parallelism["use_async"]
        enforce_eager = getattr(self, "_enforce_eager", False)

        cfg             = getattr(self, "_runner_config", {})
        max_num_seqs    = cfg.get("max_num_seqs", 512)
        gpu_memory_util = cfg.get("gpu_memory_utilization", 0.90)
        extra_kwargs    = dict(cfg.get("engine_kwargs") or {})

        # ── Filter engine_kwargs to only fields this vLLM version accepts ─────
        # Avoids TypeError when the runner config YAML references a field that
        # doesn't exist in the installed vLLM version (EngineArgs is a strict
        # dataclass — unknown keyword arguments raise TypeError immediately).
        try:
            import dataclasses
            from vllm.engine.arg_utils import EngineArgs as _EngineArgs
            _valid = {f.name for f in dataclasses.fields(_EngineArgs)}
            _dropped = {k: v for k, v in extra_kwargs.items() if k not in _valid}
            if _dropped:
                print(f"  Warning: engine_kwargs keys not supported by this "
                      f"vLLM version and will be ignored: {list(_dropped)}")
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if k in _valid}
        except Exception:
            pass  # If introspection fails, pass kwargs as-is and let vLLM report the error

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

        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )

        base_kwargs = dict(
            model=model_path,
            dtype=dtype,
            tensor_parallel_size=tp_size,
            trust_remote_code=False,
            enforce_eager=enforce_eager,
        )
        if ep_size > 1:
            base_kwargs["enable_expert_parallel"] = True
        if quantization:
            base_kwargs["quantization"] = quantization
        if max_model_len:
            base_kwargs["max_model_len"] = max_model_len

        if not use_async:
            # engine_kwargs values override named fields above if the same key appears in both.
            # This is intentional — engine_kwargs is the power-user escape hatch.
            self.llm = LLM(**{**base_kwargs, "max_num_seqs": max_num_seqs,
                              "gpu_memory_utilization": gpu_memory_util, **extra_kwargs})
        else:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            # engine_kwargs values override named fields above if the same key appears in both.
            # This is intentional — engine_kwargs is the power-user escape hatch.
            engine_args = AsyncEngineArgs(**{**base_kwargs,
                                             "gpu_memory_utilization": gpu_memory_util,
                                             **extra_kwargs})
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def get_effective_dtype(self) -> Optional[str]:
        """Report the actual compute dtype vLLM resolved after model loading."""
        try:
            if self.llm is not None:
                return str(self.llm.llm_engine.model_config.dtype).replace("torch.", "")
            elif self.engine is not None:
                return str(self.engine.engine.model_config.dtype).replace("torch.", "")
        except Exception:
            pass
        return getattr(self, "_effective_dtype", None)

    def inference_fn_offline(
        self, requests: list[InferenceRequest]
    ) -> list[InferenceResult]:
        """
        Synchronous batch inference via vLLM LLM.generate().

        ROCm: torch.cuda.* is aliased to HIP — no changes needed.
        total_time_ms is wall-clock elapsed for the full batch.
        """
        formatted = [self._format_prompt(r.prompt) for r in requests]
        t_start   = time.perf_counter()
        outputs   = self.llm.generate(formatted, self.sampling_params)
        elapsed   = time.perf_counter() - t_start

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

    async def inference_fn_streaming(
        self, request: InferenceRequest
    ) -> InferenceResult:
        """
        Async streaming via AsyncLLMEngine for TTFT measurement.
        API identical to NVIDIA vLLM runner.
        """
        from vllm.utils import random_uuid

        formatted           = self._format_prompt(request.prompt)
        request_id          = random_uuid()
        t_start             = time.perf_counter()
        first_token_time_ms = None
        output_tokens       = 0
        output_text         = ""

        async for output in self.engine.generate(
            formatted, self.sampling_params, request_id
        ):
            if (
                first_token_time_ms is None
                and len(output.outputs[0].token_ids) > 0
            ):
                first_token_time_ms = (time.perf_counter() - t_start) * 1000
            output_tokens = len(output.outputs[0].token_ids)
            output_text   = output.outputs[0].text

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
        """Async generator yielding text deltas for serve-layer SSE streaming."""
        from vllm.utils import random_uuid

        formatted   = self._format_prompt(request.prompt)
        request_id  = random_uuid()
        prev_length = 0

        async for output in self.engine.generate(
            formatted, self.sampling_params, request_id
        ):
            current_text = output.outputs[0].text
            delta        = current_text[prev_length:]
            if delta:
                yield delta
                prev_length = len(current_text)

    def get_peak_memory_gb(self) -> Optional[float]:
        """
        Query peak GPU memory via torch.cuda (aliased to HIP on ROCm).

        torch.cuda.max_memory_allocated() works on ROCm without modification
        because vLLM's ROCm build aliases the CUDA memory API to HIP.
        """
        try:
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            return None

    def release_resources(self) -> None:
        """
        Release vLLM engines and RCCL distributed state.

        The teardown sequence is identical to the NVIDIA vLLM runner.
        RCCL process groups are destroyed via the same torch.distributed API.
        """
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

        # vLLM ROCm distributed cleanup.
        # cleanup_dist_env_and_memory() handles both RCCL and Gloo teardown.
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

        # Final guard: destroy process group if still initialized.
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

        import gc
        gc.collect()
        try:
            # torch.cuda.empty_cache() works on ROCm via HIP alias
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def parse_args(self):
        """Add vLLM-ROCm/AMD-specific CLI flags. Base class pre-loads runner config."""
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

        # Priority: CLI flag > yaml config > required_chips > auto-detected > default 1
        # Fully resolved by base class.
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
            print(f"Warning: {self.__class__.__name__} does not support multi-chip. Using 1.")
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
        """Forward vLLM-ROCm/AMD-specific flags to subprocess invocations."""
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
    AMDVLLMROCmRunner().main()