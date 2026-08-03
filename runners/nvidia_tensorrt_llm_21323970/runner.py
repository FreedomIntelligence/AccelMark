"""
AccelMark — NVIDIA TensorRT-LLM benchmark script.

Implements BenchmarkRunner for TensorRT-LLM (PyTorch backend).
Uses the high-level ``tensorrt_llm.LLM`` / ``AsyncLLM`` API which mirrors
vLLM's API surface — no explicit engine build step required.

All orchestration logic lives in runners/benchmark_runner.py.
"""

import sys
from pathlib import Path

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ── MPI workaround — MUST be imported BEFORE tensorrt_llm ────────────────
# TRT-LLM calls MPI_Init at module-import time.  On containerised systems
# with a broken OpenMPI install this can crash.  trtllm_env.py (in this
# same directory) sets the required env vars.  Kept as a separate file so
# MPI-tuning changes don't affect the runner hash.
_RUNNER_DIR = Path(__file__).resolve().parent
if str(_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNNER_DIR))
import trtllm_env  # noqa: F401, E402 — side-effects only; must precede tensorrt_llm

import asyncio
import os
import time
from typing import Optional

import torch
from tensorrt_llm import LLM, SamplingParams
from transformers import AutoTokenizer

from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
from loadgen.types import InferenceResult

# Suppress verbose TRT-LLM logs during benchmark runs
import logging
logging.getLogger("tensorrt_llm").setLevel(logging.WARNING)


class TRTLLMRunner(BenchmarkRunner):
    """AccelMark benchmark runner using NVIDIA TensorRT-LLM (PyTorch backend).

    The PyTorch backend uses ``torch.export`` + ``torch.compile`` for
    automatic graph capture — no ``convert_checkpoint.py`` or
    ``trtllm-build`` step is needed.  Model loading is comparable to
    vLLM in both speed and API surface.

    In TRT-LLM 0.21, ``LLM`` provides both sync (``generate()``) and
    async (``generate_async()``) methods — no separate AsyncLLM class.
    """

    SUPPORTS_STREAMING = True
    SUPPORTS_BATCHING = True
    SUPPORTS_ONLINE = True
    SUPPORTS_MULTI_CHIP = True

    SUPPORTED_PRECISIONS = ["bf16", "fp16", "fp32"]
    # TensorRT-LLM supports FP8 natively (not emulated) on H100+ hardware.
    # On Ampere (A100/A800), FP8 falls back to BF16 with a warning.
    SUPPORTED_QUANTIZATION_BACKENDS = ["fp8"]

    def __init__(self):
        self.llm: LLM = None
        self.tokenizer: AutoTokenizer = None
        self.sampling_params: SamplingParams = None
        self._loop: asyncio.AbstractEventLoop = None

    # ── Metadata ────────────────────────────────────────────────────────────

    def _get_chip_count(self) -> int:
        try:
            n = torch.cuda.device_count()
            return n if n > 0 else 1
        except Exception:
            return 1

    def _get_framework_name(self) -> str:
        return "TensorRT-LLM"

    def _get_framework_version(self) -> str:
        try:
            import tensorrt_llm
            return tensorrt_llm.__version__
        except Exception:
            return "unknown"

    # ── Model loading ──────────────────────────────────────────────────────

    def load_model(self, model_path: str, parallelism: dict) -> None:
        """Load model — sync LLM for offline/accuracy, AsyncLLM for streaming."""
        tp_size = parallelism["tensor_parallel_size"]
        pp_size = parallelism["pipeline_parallel_size"]
        ep_size = parallelism.get("expert_parallel_size", 1)
        assert pp_size <= 1, "Pipeline parallelism is not supported in TRTLLMRunner"

        max_tokens    = parallelism["max_tokens"]
        max_model_len = parallelism["max_model_len"]
        use_async     = parallelism["use_async"]

        cfg              = getattr(self, "_runner_config", {})
        max_batch_size   = cfg.get("max_batch_size", 512)
        max_num_tokens   = cfg.get("max_num_tokens", 8192)
        extra_kwargs     = dict(cfg.get("engine_kwargs") or {})

        # ── Resolve precision / dtype ──────────────────────────────────────
        effective_precision = getattr(self, "_effective_precision", "BF16").upper()
        precision           = getattr(self, "_precision", None) or effective_precision

        _dtype_override  = getattr(self, "_precision_dtype_override", None)
        _prec_eng_kwargs = dict(getattr(self, "_precision_engine_kwargs", None) or {})

        quantization = _prec_eng_kwargs.pop("quantization", None)

        _NATIVE_DTYPE_MAP = {
            "BF16": "bfloat16",
            "FP16": "float16",
            "FP32": "float32",
        }
        dtype = _NATIVE_DTYPE_MAP.get(precision, "auto")
        self._quantization_method = quantization

        if _dtype_override:
            dtype = _dtype_override

        if _prec_eng_kwargs:
            _prec_eng_kwargs.update(extra_kwargs)
            extra_kwargs = _prec_eng_kwargs

        print(f"Loading model: precision={precision}, dtype={dtype}"
              + (f", quantization={self._quantization_method}"
                 if self._quantization_method else ""))

        # ── Tokenizer ──────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )
        # Set padding side to left for batch generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )

        # ── Build engine kwargs ────────────────────────────────────────────
        # TRT-LLM LLM constructor differs slightly from vLLM:
        #   - No ``gpu_memory_utilization`` → use ``free_gpu_memory_fraction``
        #   - No ``enforce_eager`` → always uses torch.compile (CUDA graphs)
        #   - ``max_batch_size`` + ``max_num_tokens`` instead of ``max_num_seqs``
        #   - No ``trust_remote_code`` parameter
        #   - ``backend="pytorch"`` (default) for graph-capture path
        engine_kwargs = dict(
            model=model_path,
            dtype=dtype,
            tensor_parallel_size=tp_size,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            backend="pytorch",
            **extra_kwargs,
        )
        if ep_size > 1:
            engine_kwargs["moe_expert_parallel_size"] = ep_size
        if max_model_len:
            engine_kwargs["max_seq_len"] = max_model_len
        if quantization:
            engine_kwargs["quant_config"] = quantization

        # TRT-LLM 0.21: one LLM instance supports both sync generate()
        # and async generate_async().  No separate AsyncLLM class.
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.llm = LLM(**engine_kwargs)

    def get_effective_dtype(self) -> Optional[str]:
        """Report the actual compute dtype TRT-LLM resolved after model loading."""
        try:
            if self.llm is not None:
                return str(self.llm.dtype).replace("torch.", "")
        except Exception:
            pass
        return getattr(self, "_effective_dtype", None)

    # ── Offline (batch) inference ──────────────────────────────────────────

    def inference_fn_offline(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        """Send all requests at once through TRT-LLM for maximum throughput.

        total_time_ms is the wall-clock elapsed time of the entire batch,
        shared across all results.  LoadGen divides total output tokens by
        this elapsed time to compute throughput.
        """
        formatted = [self._format_prompt(r.prompt) for r in requests]
        t_start = time.perf_counter()
        outputs = self.llm.generate(formatted, self.sampling_params)
        elapsed = time.perf_counter() - t_start

        # Store output text for accuracy scenario
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

    # ── Streaming inference ────────────────────────────────────────────────

    async def inference_fn_streaming(self, request: InferenceRequest) -> InferenceResult:
        """Stream a single request through AsyncLLM, measuring TTFT."""
        formatted = self._format_prompt(request.prompt)
        t_start = time.perf_counter()
        first_token_time_ms = None
        output_tokens = 0
        output_text = ""

        async for output in self.llm.generate_async(
            formatted, self.sampling_params, streaming=True
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
        """Yield decoded text deltas for SSE streaming in serve mode.

        TRT-LLM's generate_async yields cumulative outputs, so we slice
        off only the new text since the previous iteration.
        """
        formatted   = self._format_prompt(request.prompt)
        prev_length = 0

        async for output in self.llm.generate_async(
            formatted, self.sampling_params, streaming=True
        ):
            current_text = output.outputs[0].text
            delta = current_text[prev_length:]
            if delta:
                yield delta
                prev_length = len(current_text)

    # ── Resource cleanup ───────────────────────────────────────────────────

    def get_peak_memory_gb(self) -> Optional[float]:
        try:
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            return None

    def release_resources(self) -> None:
        """Release TRT-LLM engine and distributed state."""
        if self.llm is not None:
            try:
                del self.llm
            except Exception:
                pass
            self.llm = None

        # Destroy any lingering torch distributed state so the next engine
        # initialisation creates a fresh process group.
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

        torch.cuda.empty_cache()

    # ── CLI flags ──────────────────────────────────────────────────────────

    def parse_args(self):
        """Add TRT-LLM-specific CLI flags."""
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

        # TRT-LLM PyTorch backend always uses torch.compile / CUDA graphs —
        # enforce_eager is accepted for CLI compatibility but has no effect.
        if extra.enforce_eager:
            print("  Note: --enforce-eager has no effect on TRT-LLM PyTorch backend")

        print(f"  tensor_parallel_size = {tp_size}  [{_tp_source}]")
        if ep_size > 1:
            print(f"  expert_parallel_size = {ep_size}")

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
        """Forward TRT-LLM-specific flags to scenario subprocesses."""
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
        return extra


if __name__ == "__main__":
    TRTLLMRunner().main()
