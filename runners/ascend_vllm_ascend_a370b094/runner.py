"""
AccelMark — Huawei Ascend NPU benchmark runner (vllm-ascend).

Implements BenchmarkRunner for vllm-ascend on Huawei Ascend NPUs.
All orchestration logic lives in runners/benchmark_runner.py.

vllm-ascend keeps the standard vLLM Python API (LLM, AsyncLLMEngine,
SamplingParams) while replacing the CUDA backend with CANN. This runner
is therefore structurally identical to the NVIDIA vLLM runner — the
differences are in capability flags, precision/quantization mapping, and
NPU-specific memory and process group teardown.

Hardware:     Huawei Ascend 910B / 910C NPU series
Runtime:      CANN (Compute Architecture for Neural Networks)
Framework:    vllm-ascend — https://github.com/vllm-project/vllm-ascend
Precision:    BF16 (preferred), FP16 (fallback). FP8 not supported on
              current Ascend 910B/910C hardware.
Quantization: W8A8, W8A16, W4A16 via vllm-ascend quantization layer.
Multi-chip:   Tensor parallelism via HCCL.
Streaming:    Fully supported — AsyncLLMEngine API is identical to vLLM.

Installation:
    # 1. Install CANN toolkit matching your NPU driver version:
    #    https://www.hiascend.com/software/cann
    # 2. Install torch_npu (Huawei PyTorch NPU extension):
    #    https://gitee.com/ascend/pytorch
    # 3. Install vllm-ascend and runner dependencies:
    pip install -r runners/ascend_vllm_ascend_{hash8}/requirements.txt

Usage:
    python run.py --runner ascend_vllm_ascend_{hash8} --suite suite_A
"""

import asyncio
import gc
import sys
import time
from pathlib import Path
from typing import Optional

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
from loadgen.types import InferenceResult



import logging
logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)
logging.getLogger("vllm.engine.llm_engine").setLevel(logging.WARNING)


class AscendVLLMRunner(BenchmarkRunner):
    """
    AccelMark benchmark runner using vllm-ascend on Huawei Ascend NPUs.

    vllm-ascend preserves the standard vLLM API (LLM, AsyncLLMEngine,
    SamplingParams) while replacing the CUDA kernel layer with CANN.
    The inference methods are therefore identical to the NVIDIA vLLM runner.
    Platform-specific behaviour is isolated to load_model(), get_peak_memory_gb(),
    and release_resources().
    """

    SUPPORTS_STREAMING  = True
    SUPPORTS_BATCHING   = True
    SUPPORTS_ONLINE     = True
    SUPPORTS_MULTI_CHIP = True  # HCCL-based tensor parallelism

    # Ascend 910B/910C supports BF16 natively for LLM workloads.
    # FP16 is available as a fallback on older or constrained configs.
    # FP8 is not supported on current Ascend hardware — excluded entirely.
    SUPPORTED_PRECISIONS = ["bf16", "fp16", "fp32"]

    # W8A8, W8A16, W4A16 are supported via vllm-ascend's quantization layer.
    # FP8 is excluded — no native FP8 hardware support on Ascend 910B/910C.
    SUPPORTED_QUANTIZATIONS = ["w8a8", "w8a16", "w4a16"]

    def __init__(self):
        self.llm             = None  # vllm.LLM (offline / accuracy)
        self.engine          = None  # vllm.AsyncLLMEngine (online / interactive)
        self.tokenizer       = None
        self.sampling_params = None
        self._loop: asyncio.AbstractEventLoop = None

    def _get_chip_count(self) -> int:
        """Return the number of available Ascend NPUs, falling back to CUDA."""
        try:
            import torch_npu
            n = torch_npu.npu.device_count()
            if n > 0:
                return n
        except Exception:
            pass
        try:
            import torch
            n = torch.cuda.device_count()
            return n if n > 0 else 1
        except Exception:
            return 1

    def _get_framework_name(self) -> str:
        return "vllm-ascend"

    def _get_framework_version(self) -> str:
        try:
            import vllm
            return vllm.__version__
        except Exception:
            return "unknown"

    def get_model_format(self) -> str:
        return "HuggingFace original"

    def load_model(self, model_path: str, parallelism: dict) -> None:
        """
        Load model onto Ascend NPU via vllm-ascend.

        vllm-ascend uses the standard vLLM LLM / AsyncLLMEngine constructors.
        The CANN backend activates automatically when vllm-ascend is installed
        and Ascend NPUs are present — no explicit device flag is required in
        the engine kwargs; vllm-ascend patches the vLLM device selection layer
        at import time.

        Pipeline parallelism is not supported in this runner (same limitation
        as the vLLM CUDA backend). Use tensor_parallel_size for multi-chip runs.
        """
        from transformers import AutoTokenizer
        from vllm import LLM, AsyncLLMEngine, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs

        tp_size       = parallelism["tensor_parallel_size"]
        pp_size       = parallelism["pipeline_parallel_size"]
        assert pp_size <= 1, (
            "Pipeline parallelism (pp_size > 1) is not supported in "
            "AscendVLLMRunner. Use --tensor-parallel-size for multi-chip runs."
        )

        max_tokens    = parallelism["max_tokens"]
        max_model_len = parallelism["max_model_len"]
        use_async     = parallelism["use_async"]
        enforce_eager = getattr(self, "_enforce_eager", False)

        cfg          = getattr(self, "_runner_config", {})
        max_num_seqs = cfg.get("max_num_seqs", 512)
        extra_kwargs = cfg.get("engine_kwargs") or {}

        effective_precision = getattr(self, "_effective_precision", "BF16").upper()

        # Map AccelMark precision names to vllm-ascend dtype.
        # Pre-quantized checkpoints use dtype="auto" so vllm-ascend reads
        # the quantization config directly from the checkpoint's config.json.
        dtype = "bfloat16"

        if effective_precision == "BF16":
            dtype = "bfloat16"
            self._quantization_method = None
        elif effective_precision == "FP16":
            dtype = "float16"
            self._quantization_method = None
        elif effective_precision == "FP32":
            dtype = "float32"
            self._quantization_method = None
        elif effective_precision == "W8A8":
            dtype = "auto"
            self._quantization_method = "w8a8"
        elif effective_precision == "W8A16":
            dtype = "auto"
            self._quantization_method = "w8a16"
        elif effective_precision == "W4A16":
            dtype = "auto"
            self._quantization_method = "awq"
        else:
            dtype = "auto"
            self._quantization_method = None

        print(
            f"Loading model: precision={effective_precision}, dtype={dtype}"
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
        if max_model_len:
            base_kwargs["max_model_len"] = max_model_len

        if not use_async:
            # engine_kwargs values override named fields above if the same key appears in both.
            # This is intentional — engine_kwargs is the power-user escape hatch.
            self.llm = LLM(**{**base_kwargs, "max_num_seqs": max_num_seqs, **extra_kwargs})
        else:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            # engine_kwargs values override named fields above if the same key appears in both.
            # This is intentional — engine_kwargs is the power-user escape hatch.
            engine_args = AsyncEngineArgs(**{**base_kwargs, **extra_kwargs})
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def get_effective_dtype(self) -> Optional[str]:
        """Report the actual compute dtype vllm-ascend resolved after model loading."""
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
        Synchronous batch inference via vllm-ascend LLM.generate().
        total_time_ms is wall-clock elapsed time for the full batch.
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
        Async streaming via vllm-ascend AsyncLLMEngine for TTFT measurement.
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
        """Async generator yielding decoded text deltas for serve-layer SSE streaming."""
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
        Query peak NPU memory usage via torch_npu.

        torch_npu is Huawei's PyTorch extension for Ascend NPUs and provides
        npu.max_memory_allocated() mirroring the torch.cuda equivalent.
        Returns None if torch_npu is not installed.
        """
        try:
            import torch_npu
            return torch_npu.npu.max_memory_allocated() / (1024 ** 3)
        except Exception:
            return None

    def release_resources(self) -> None:
        """
        Release vllm-ascend engines and NPU memory.

        Teardown order:
          1. Shut down async engine (if online/interactive was used)
          2. Delete engine objects to trigger Python GC
          3. vllm-ascend distributed state cleanup via vLLM's standard API
          4. HCCL / torch.distributed process group destruction
          5. NPU memory cache flush via torch_npu
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

        # vllm-ascend distributed state cleanup.
        # cleanup_dist_env_and_memory() is the same entry point as standard vLLM —
        # vllm-ascend patches the internals but keeps the public function name.
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

        # Destroy HCCL process group via torch.distributed.
        # On Ascend, torch.distributed uses HCCL as the backend when
        # initialized by vllm-ascend — the destroy API is identical to NCCL.
        try:
            import torch
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

        gc.collect()

        # Flush NPU memory cache.
        # torch_npu.npu.empty_cache() releases cached but unused NPU memory
        # back to the CANN allocator — equivalent to torch.cuda.empty_cache().
        try:
            import torch_npu
            torch_npu.npu.empty_cache()
            torch_npu.npu.reset_peak_memory_stats()
        except Exception:
            pass

    def parse_args(self):
        """Add vllm-ascend/Ascend-specific CLI flags. Base class pre-loads runner config."""
        args = super().parse_args()
        cfg = self._runner_config

        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--tensor-parallel-size", type=int, default=None,
                            dest="tensor_parallel_size")
        parser.add_argument("--enforce-eager", action="store_true", default=False,
                            dest="enforce_eager")
        extra, _ = parser.parse_known_args()

        # Priority: CLI flag > yaml config > required_chips > auto-detected > default 1
        # Fully resolved by base class.
        tp_size, _tp_source = self._resolve_tensor_parallel_size(
            extra.tensor_parallel_size
        )

        self._enforce_eager = extra.enforce_eager or cfg.get("enforce_eager", False)

        print(f"  tensor_parallel_size = {tp_size}  [{_tp_source}]")

        self._parallelism = {
            "tensor_parallel_size":   tp_size,
            "pipeline_parallel_size": 1,
            "expert_parallel_size":   1,
            "data_parallel_size":     1,
        }
        self._chip_count = tp_size
        return args

    def get_extra_subprocess_args(self, args) -> list[str]:
        """Forward vllm-ascend/Ascend-specific flags to subprocess invocations."""
        extra = [
            "--tensor-parallel-size",
            str(self._parallelism.get("tensor_parallel_size", 1)),
        ]
        if self._enforce_eager:
            extra += ["--enforce-eager"]
        return extra


if __name__ == "__main__":
    AscendVLLMRunner().main()
