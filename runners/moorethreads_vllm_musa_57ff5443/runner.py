"""
AccelMark — Moore Threads MUSA GPU benchmark runner (vllm-musa).

Implements BenchmarkRunner for vLLM on Moore Threads MUSA GPUs via the
``vllm-musa`` platform plugin. All orchestration logic lives in
``runners/benchmark_runner.py``.

The plugin works by patching vLLM at import time:
  - ``torchada`` aliases the CUDA Python API onto MUSA
  - ``pymtml`` (mthreads-ml-py) provides device queries equivalent to
    nvidia-ml-py
  - A few Triton attention/worker patches are applied to make the standard
    vLLM kernels run on MUSA's Triton compiler.

As a result, the standard vLLM Python API (``LLM``, ``AsyncLLMEngine``,
``SamplingParams``) is fully preserved. This runner is therefore structurally
identical to the NVIDIA / AMD / Ascend vLLM runners — the differences are
in capability flags, device-count detection, and memory teardown.

Hardware:     Moore Threads MTT S4000 / S5000 (and forward-compatible
              successors). S3000 / S80 may also work but are not the public
              reference target.
Runtime:      MUSA (Meta-computing Unified System Architecture)
Framework:    vllm-musa — https://github.com/MooreThreads/vllm-musa
              (also published on PyPI as ``vllm-musa``)
Precision:    BF16 (preferred on S4000+), FP16 fallback. FP8 not yet
              supported on shipping MUSA hardware.
Quantization: compressed-tensors (W8A8 / W8A16) declared by default. AWQ /
              GPTQ / FP8 may be added once validated on real hardware.
Multi-chip:   Tensor parallelism via MCCL (Moore Threads Collective
              Communications Library). vLLM's tensor_parallel_size flag works
              unchanged because torchada aliases the NCCL API surface.
Streaming:    Fully supported — AsyncLLMEngine API is identical to vLLM.

Installation (without a real device this is "informational"; final
versions to be confirmed at smoke-test time):

    # 1. Install the MUSA toolkit + driver matching your card firmware:
    #    https://developer.mthreads.com/musa/
    # 2. Install Moore Threads' PyTorch build (torch + torchada) inside the
    #    official MUSA container, then:
    pip install -r runners/moorethreads_vllm_musa_{hash8}/requirements.txt

Usage:

    # S5000 single chip
    python run.py --runner moorethreads_vllm_musa_{hash8} --suite suite_F

    # Multi-chip tensor parallelism (e.g. 8 x S5000)
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    python run.py --runner moorethreads_vllm_musa_{hash8} \
        --suite suite_B --tensor-parallel-size 8

Environment variables you might want to set:
    MUSA_VISIBLE_DEVICES        — equivalent to CUDA_VISIBLE_DEVICES
    VLLM_WORKER_MULTIPROC_METHOD=spawn   — recommended for multi-process workers
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


class MoorethreadsVLLMMUSARunner(BenchmarkRunner):
    """
    AccelMark benchmark runner using ``vllm-musa`` on Moore Threads MUSA GPUs.

    ``vllm-musa`` is registered as a vLLM platform plugin and is auto-detected
    on ``import vllm``. The plugin activates the MUSA backend when:
      - the plugin package is installed in the environment
      - Moore Threads devices are visible to the process

    The inference methods below are byte-for-byte identical in shape to the
    NVIDIA vLLM runner — platform-specific logic is isolated to
    ``_get_chip_count()``, ``load_model()``, ``get_peak_memory_gb()``, and
    ``release_resources()``.
    """

    SUPPORTS_STREAMING = True
    SUPPORTS_BATCHING = True
    SUPPORTS_ONLINE = True
    SUPPORTS_MULTI_CHIP = True  # MCCL-based tensor parallelism on multi-card hosts

    # S4000 / S5000 advertise native BF16 for LLM workloads; FP16 always works
    # as a fallback. FP32 is left in the list for completeness but is rarely
    # used for inference. FP8 is excluded entirely — current shipping MUSA
    # hardware does not expose native FP8 datapaths.
    SUPPORTED_PRECISIONS = ["bf16", "fp16", "fp32"]

    # Quantization backends — start conservative. compressed-tensors is the
    # safe default on every modern vLLM build because the kernels are pure
    # Triton + PyTorch matmuls and so are reachable through torchada.
    # Marlin / AWQ-CUDA / native FP8 require kernel-level validation on MUSA
    # and should be added in a follow-up runner version after real-hardware
    # smoke tests, not silently flipped on here.
    SUPPORTED_QUANTIZATION_BACKENDS = ["compressed-tensors"]

    def __init__(self):
        self.llm = None              # vllm.LLM (offline / accuracy)
        self.engine = None           # vllm.AsyncLLMEngine (online / interactive)
        self.tokenizer = None
        self.sampling_params = None
        self._loop: asyncio.AbstractEventLoop = None

    # ── Metadata ─────────────────────────────────────────────────────────────

    def _get_chip_count(self) -> int:
        """Return the number of available Moore Threads MUSA GPUs.

        Preference order:
          1. ``pymtml`` (the Moore Threads management library, equivalent to
             nvidia-ml-py). Most reliable because it queries the driver
             directly and is not affected by ``MUSA_VISIBLE_DEVICES`` if
             called before any ``torch`` initialisation.
          2. ``torch.cuda.device_count()`` — torchada aliases ``torch.cuda``
             to MUSA so this returns the visible MUSA device count in the
             current process (respecting ``MUSA_VISIBLE_DEVICES``).
        """
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
        # The leaderboard groups by framework string; keep this distinct from
        # plain "vLLM" so MUSA results are not silently mixed with CUDA results.
        return "vllm-musa"

    def _get_framework_version(self) -> str:
        """Report vllm-musa plugin version, with vLLM core version appended.

        The plugin version is the meaningful identifier (it pins the patch
        set), but the underlying vLLM core version is what generates kernels
        and parses configs. Reporting both makes results reproducible.
        """
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

        core_version = "unknown"
        try:
            import vllm
            core_version = vllm.__version__
        except Exception:
            pass

        if plugin_version == "unknown" and core_version == "unknown":
            return "unknown"
        if plugin_version == "unknown":
            return core_version
        return f"{plugin_version}+vllm-{core_version}"

    def get_model_format(self) -> str:
        return "HuggingFace original"

    # ── Model loading ────────────────────────────────────────────────────────

    def load_model(self, model_path: str, parallelism: dict) -> None:
        """
        Load model onto Moore Threads MUSA GPU(s) via vllm-musa.

        vllm-musa uses the standard vLLM ``LLM`` / ``AsyncLLMEngine``
        constructors. The MUSA backend activates automatically when the
        plugin package is installed and Moore Threads devices are present —
        no explicit device flag is required in engine kwargs.

        Pipeline parallelism is not supported (matches the vLLM CUDA backend
        behaviour). Use ``tensor_parallel_size`` for multi-chip runs.
        """
        from transformers import AutoTokenizer
        from vllm import LLM, AsyncLLMEngine, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs

        tp_size = parallelism["tensor_parallel_size"]
        pp_size = parallelism["pipeline_parallel_size"]
        ep_size = parallelism.get("expert_parallel_size", 1)
        assert pp_size <= 1, (
            "Pipeline parallelism (pp_size > 1) is not supported in "
            "MoorethreadsVLLMMUSARunner. Use --tensor-parallel-size for "
            "multi-chip runs."
        )

        max_tokens = parallelism["max_tokens"]
        max_model_len = parallelism["max_model_len"]
        use_async = parallelism["use_async"]
        enforce_eager = getattr(self, "_enforce_eager", False)

        cfg = getattr(self, "_runner_config", {})
        max_num_seqs = cfg.get("max_num_seqs", 256)
        # vLLM's flag name is gpu_memory_utilization, but on MUSA it controls
        # the per-card HBM fraction reserved for the KV cache. We keep the
        # vLLM name to stay schema-compatible with other runners' configs.
        musa_memory_util = cfg.get("gpu_memory_utilization", 0.85)
        extra_kwargs = dict(cfg.get("engine_kwargs") or {})

        # Filter engine_kwargs to only fields the installed vLLM version
        # accepts. EngineArgs is a strict dataclass — unknown kwargs raise
        # TypeError at construction. vllm-musa supports vLLM 0.10.x and 0.13.x,
        # whose EngineArgs fields differ slightly; filtering keeps the YAML
        # forward-compatible.
        try:
            import dataclasses
            from vllm.engine.arg_utils import EngineArgs as _EngineArgs
            _valid = {f.name for f in dataclasses.fields(_EngineArgs)}
            _dropped = {k: v for k, v in extra_kwargs.items() if k not in _valid}
            if _dropped:
                print(f"  Warning: engine_kwargs keys not supported by this "
                      f"vllm-musa / vLLM version and will be ignored: "
                      f"{list(_dropped)}")
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if k in _valid}
        except Exception:
            pass

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
        if _prec_eng_kwargs:
            _prec_eng_kwargs.update(extra_kwargs)
            extra_kwargs = _prec_eng_kwargs

        # Translate the runner's flat speculative-decoding keys into the
        # dict-form ``speculative_config`` used by recent vLLM versions. Skip
        # if the user already provided ``speculative_config`` directly.
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
        """Report the actual compute dtype vllm-musa resolved after loading."""
        try:
            if self.llm is not None:
                return str(self.llm.llm_engine.model_config.dtype).replace("torch.", "")
            elif self.engine is not None:
                return str(self.engine.engine.model_config.dtype).replace("torch.", "")
        except Exception:
            pass
        return getattr(self, "_effective_dtype", None)

    # ── Inference ────────────────────────────────────────────────────────────

    def inference_fn_offline(
        self, requests: list[InferenceRequest]
    ) -> list[InferenceResult]:
        """
        Synchronous batch inference via vllm-musa LLM.generate().
        total_time_ms is wall-clock elapsed time for the full batch.
        """
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

    async def inference_fn_streaming(
        self, request: InferenceRequest
    ) -> InferenceResult:
        """Async streaming for TTFT — API identical to NVIDIA vLLM runner."""
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
        """Async generator yielding text deltas for serve-layer SSE."""
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

    # ── Memory & teardown ────────────────────────────────────────────────────

    def get_peak_memory_gb(self) -> Optional[float]:
        """Query peak HBM usage on the active MUSA device.

        torchada aliases ``torch.cuda.max_memory_allocated()`` onto MUSA, so
        the standard CUDA API returns peak MUSA memory. We fall back to
        ``pymtml`` if torch is unavailable for some reason.
        """
        try:
            import torch
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            pass
        # pymtml fallback — returns currently-used memory, not strictly peak,
        # but useful when torch.cuda is gone.
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
        """
        Release vllm-musa engines and MUSA memory.

        Teardown order mirrors the NVIDIA runner:
          1. Shut down async engine (if online/interactive was used)
          2. Delete engine objects to trigger Python GC
          3. vLLM distributed-state cleanup (cleanup_dist_env_and_memory)
          4. MCCL / torch.distributed process group destruction
          5. MUSA memory cache flush via torch.cuda (aliased to MUSA by torchada)
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

        # vLLM distributed state cleanup. cleanup_dist_env_and_memory is the
        # same entry point as upstream vLLM — vllm-musa patches the internals
        # but keeps the public function name.
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

        # Destroy the active torch.distributed process group. On MUSA the
        # backend is MCCL (Moore Threads Collective Communications Library)
        # but is exposed through the standard torch.distributed.destroy_process_group
        # entry point thanks to torchada.
        try:
            import torch
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

        gc.collect()

        # Flush MUSA memory cache. torch.cuda.* is aliased to MUSA by torchada,
        # so the standard CUDA cache-management APIs work without modification.
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    # ── Argument parsing ─────────────────────────────────────────────────────

    def parse_args(self):
        """Add vllm-musa / Moore Threads-specific CLI flags."""
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

        self._parallelism = {
            "tensor_parallel_size":   tp_size,
            "pipeline_parallel_size": 1,
            "expert_parallel_size":   ep_size,
            "data_parallel_size":     1,
        }
        self._chip_count = tp_size
        return args

    def get_extra_subprocess_args(self, args) -> list[str]:
        """Forward vllm-musa / Moore Threads-specific flags to subprocesses."""
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
