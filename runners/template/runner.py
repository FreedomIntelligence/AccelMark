"""
AccelMark Runner Template

Copy this file to runners/{platform}_{framework}_{hash8}/runner.py
and implement the TODO sections. Then compute the hash:

    python runners/hash_runner.py runners/your_folder/runner.py

Rename your folder to the printed ID before submitting a PR.

See runners/README.md for the full submission guide.
See DEVELOPMENT.md for a complete worked example (LMDeploy).
"""

import sys
import time
from pathlib import Path
from typing import Optional

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
from loadgen.types import InferenceResult


class TemplateRunner(BenchmarkRunner):
    """
    Replace 'TemplateRunner' with your class name, e.g. 'SGLangRunner'.
    """

    # ── Capability flags ──────────────────────────────────────────────────────

    SUPPORTS_STREAMING = True
    """
    True if your framework has a token-streaming API (async generator).
    Required for accurate TTFT measurement in online/interactive/sustained.
    Set False if your framework only supports full-response inference —
    TTFT will not be measured and online/interactive/sustained will be skipped.
    """

    SUPPORTS_BATCHING = True
    """
    True if your framework can process multiple requests simultaneously.
    Set False for serial-only frameworks (e.g. mlx-lm) — offline scenario
    will send one request at a time instead of a full batch.
    """

    SUPPORTS_ONLINE = True
    """
    True if your framework can handle concurrent requests.
    Set False if the inference API is single-threaded.
    """

    SUPPORTS_MULTI_CHIP = True
    """
    True if your framework supports tensor parallelism across multiple GPUs.
    Set False if single-GPU only — --tensor-parallel-size is ignored.
    """

    SUPPORTED_PRECISIONS = ["bf16", "fp16", "fp32"]
    """
    Compute precisions your framework supports on capable hardware.
    BenchmarkRunner auto-detects hardware limits and intersects with this list.
    """

    SUPPORTED_QUANTIZATIONS = []
    """
    Quantization formats for Suite C. List any of: "fp8", "w8a8", "w8a16", "w4a16"
    BF16 is always supported — do not list it here.
    Empty list = this runner skips all quantized formats in Suite C.
    """

    # ── Initializer ───────────────────────────────────────────────────────────

    def __init__(self):
        # TODO: declare instance variables for your model/engine/tokenizer
        self.model     = None
        self.tokenizer = None

    # ── Required: load model ──────────────────────────────────────────────────

    def load_model(self, model_path: str, parallelism: dict) -> None:
        """
        Load model weights into accelerator memory.

        Args:
            model_path:  Resolved local path or HuggingFace model ID.
            parallelism: Engine configuration dict. Always contains:
                             "tensor_parallel_size":   int   (default 1)
                             "pipeline_parallel_size": int   (default 1)
                             "expert_parallel_size":   int   (default 1)
                             "data_parallel_size":     int   (default 1)
                             "max_tokens":             int   — max generation tokens
                             "max_model_len":          int|None — context window limit
                             "use_async":              bool  — True for online/interactive/sustained
                         Read only the keys you need. Ignore unknown keys.
        """
        # TODO: load your model
        # Example:
        #   tp_size = parallelism["tensor_parallel_size"]
        #   max_tokens = parallelism["max_tokens"]
        #   self.model = MyFramework.load(model_path, tp=tp_size)
        raise NotImplementedError

    # ── Required: offline batch inference ─────────────────────────────────────

    def inference_fn_offline(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        """
        Synchronous batch inference. Send all requests at once.

        Args:
            requests: List of InferenceRequest objects.
                      Read request.prompt for the formatted prompt string.
                      Other fields (max_tokens, temperature) are optional.

        Returns:
            List[InferenceResult] — same length and order as requests.

        Note: total_time_ms should be the wall-clock elapsed time for the
        entire batch, shared across all results. LoadGen uses it to compute
        throughput = total_tokens / (elapsed_ms / 1000).
        """
        # TODO: run batch inference
        # Example:
        #   prompts = [self._format_prompt(r.prompt) for r in requests]
        #   t_start = time.perf_counter()
        #   outputs = self.model.generate(prompts)
        #   elapsed_ms = (time.perf_counter() - t_start) * 1000
        #   return [
        #       InferenceResult(
        #           first_token_time_ms=None,
        #           total_time_ms=elapsed_ms,
        #           output_tokens=len(o.token_ids),
        #           input_tokens=len(o.input_token_ids),
        #           success=True,
        #       )
        #       for o in outputs
        #   ]
        raise NotImplementedError

    # ── Required: resource cleanup ─────────────────────────────────────────────

    def release_resources(self) -> None:
        """
        Release accelerator memory and any distributed process groups.
        Must be safe to call multiple times (if self.model is None, just return).
        """
        # TODO: release your model
        # Example:
        #   if self.model is not None:
        #       del self.model
        #       self.model = None
        pass

    # ── Optional: streaming inference ─────────────────────────────────────────

    async def inference_fn_streaming(self, request: InferenceRequest) -> InferenceResult:
        """
        Async single-request streaming inference for online/interactive scenarios.
        Required when SUPPORTS_STREAMING = True (the default).

        Args:
            request: InferenceRequest. Read request.prompt at minimum.

        Returns:
            InferenceResult with first_token_time_ms set for TTFT measurement.
        """
        # TODO: implement streaming inference
        # Example:
        #   formatted = self._format_prompt(request.prompt)
        #   t_start = time.perf_counter()
        #   first_token_time_ms = None
        #   output_text = ""
        #   output_tokens = 0
        #
        #   async for token in self.model.stream(formatted):
        #       if first_token_time_ms is None:
        #           first_token_time_ms = (time.perf_counter() - t_start) * 1000
        #       output_text   += token
        #       output_tokens += 1
        #
        #   return InferenceResult(
        #       first_token_time_ms=first_token_time_ms,
        #       total_time_ms=(time.perf_counter() - t_start) * 1000,
        #       output_tokens=output_tokens,
        #       input_tokens=0,
        #       success=True,
        #       output_text=output_text,
        #   )
        raise NotImplementedError(
            f"{self.__class__.__name__} sets SUPPORTS_STREAMING=True "
            f"but does not implement inference_fn_streaming(). "
            f"Either implement it or set SUPPORTS_STREAMING = False."
        )

    # ── Optional: token streaming for serve layer ─────────────────────────────

    async def inference_fn_token_stream(self, request: InferenceRequest):
        """
        Async generator yielding text deltas for `run.py --serve` SSE streaming.
        Optional — serve layer falls back to inference_fn_streaming() if not
        implemented.

        Yields:
            str — decoded text delta (NOT cumulative).
        """
        # TODO (optional): yield tokens one at a time
        # Example (for frameworks with cumulative output like vLLM):
        #   formatted = self._format_prompt(request.prompt)
        #   prev_len = 0
        #   async for output in self.engine.generate(formatted, ...):
        #       delta = output.text[prev_len:]
        #       if delta:
        #           yield delta
        #           prev_len = len(output.text)
        raise NotImplementedError
        if False:
            yield  # makes this an async generator function

    # ── Optional: metadata ────────────────────────────────────────────────────

    def _get_framework_name(self) -> str:
        """Return the framework name for result.json. e.g. 'vLLM', 'SGLang'."""
        # TODO: return your framework name
        return "MyFramework"

    def _get_framework_version(self) -> str:
        """Return the framework version string."""
        # TODO: return your framework version
        try:
            import myframework
            return myframework.__version__
        except Exception:
            return "unknown"

    def format_prompt(self, prompt: str) -> str:
        """
        Apply chat template or other prompt formatting.
        Override if your tokenizer has a chat template to apply.
        Default: return prompt unchanged.
        """
        # TODO (optional): apply chat template
        # Example:
        #   if self.tokenizer and self.tokenizer.chat_template:
        #       return self.tokenizer.apply_chat_template(
        #           [{"role": "user", "content": prompt}],
        #           tokenize=False,
        #           add_generation_prompt=True,
        #       )
        return prompt

    def get_peak_memory_gb(self) -> Optional[float]:
        """
        Return peak accelerator memory usage in GB after inference.
        Default: returns None (not measured).
        """
        # TODO (optional): query peak memory
        # NVIDIA example:
        #   import torch
        #   return torch.cuda.max_memory_allocated() / (1024 ** 3)
        return None

    def get_effective_dtype(self) -> Optional[str]:
        """
        Return the actual compute dtype used after load_model().
        Override when it may differ from requested precision
        (e.g. FP8 weights on A100 compute in bfloat16).
        """
        return getattr(self, "_effective_dtype", None)

    def get_extra_subprocess_args(self, args) -> list[str]:
        """
        Return extra CLI args to forward to scenario subprocesses.
        Override if parse_args() adds custom arguments.
        Example:
            def get_extra_subprocess_args(self, args):
                extra = []
                if getattr(args, "max_num_seqs", None):
                    extra += ["--max-num-seqs", str(args.max_num_seqs)]
                return extra
        """
        return []


if __name__ == "__main__":
    TemplateRunner().main()