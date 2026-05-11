"""
AccelMark — Google TPU benchmark runner (vllm-tpu / tpu-inference).

Implements BenchmarkRunner for vllm-tpu on Google Cloud TPUs.
All orchestration logic lives in runners/benchmark_runner.py.

The vllm-tpu backend (tpu-inference) is a JAX/XLA-based plugin for vLLM
developed by Google. It uses the standard vLLM Python API (LLM, SamplingParams)
but compiles models with XLA rather than CUDA graphs.

Key TPU-specific behaviours this runner handles:

  1. BF16 only — v5e natively supports BF16; FP16 is not a first-class
     dtype for LLM inference on TPU. SUPPORTED_PRECISIONS = ["bf16"].

  2. No async streaming engine — vllm-tpu currently does not expose
     AsyncLLMEngine / token-by-token streaming (is_async_output_supported
     returns False in the platform layer). SUPPORTS_STREAMING = False.
     Online / interactive / sustained scenarios are therefore skipped.

  3. XLA compilation warmup — first run compiles XLA graphs per input shape,
     taking 20–30 minutes. Subsequent runs reuse the on-disk cache.
     On Colab (ephemeral sessions), persist VLLM_XLA_CACHE_PATH to
     Google Drive to avoid recompiling every session.

  4. Chip detection via /dev/accel* — tpu_info.get_num_chips() counts
     /dev/accel* device files (1 on v5e-1 Colab).

  5. Memory — v5e-1 has 16 GiB HBM per chip. Llama-3-8B in BF16 uses
     ~16 GB, leaving almost no KV cache headroom. Use a small max_model_len
     (≤2048) and max_num_seqs=1. Suite F (Qwen2.5-0.5B) is comfortable.

  6. Model support — tpu-inference JAX-native registry:
     tpu-inference compiles models via JAX/XLA. Models not in the native registry
     fall back to PyTorch/torchax, which hits a recursive JIT error on v5e.

     Supported (JAX-native, confirmed working on v5e):
       LlamaForCausalLM   → meta-llama/Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct
       Qwen3ForCausalLM   → Qwen/Qwen3-0.6B, Qwen/Qwen3-4B, Qwen/Qwen3-32B

     NOT supported (falls back to PyTorch, crashes with RecursionError on v5e):
       Qwen2ForCausalLM   → Qwen/Qwen2.5-0.5B-Instruct, Qwen2.5-7B-Instruct, etc.

     Suite F uses Qwen2.5-0.5B-Instruct (Qwen2ForCausalLM) — it will not run on
     this runner until tpu-inference adds Qwen2ForCausalLM to its JAX model registry.
     Use Qwen/Qwen3-0.6B as a substitute for smoke testing.

Hardware:     Google TPU v5e (single chip, v5e-1)
Runtime:      JAX/XLA via tpu-inference plugin
Framework:    vllm-tpu — https://github.com/vllm-project/tpu-inference
Precision:    BF16 only
Quantization: None supported on v5e for benchmarking (untested on hardware)
Multi-chip:   Not applicable for v5e-1 (single chip)
Streaming:    Not supported (is_async_output_supported = False in platform)

Installation (Colab / TPU VM):
    # Install vllm-tpu (includes tpu-inference plugin and JAX/XLA)
    pip install vllm-tpu

    # Optional: persist XLA cache across Colab sessions (avoids 20-30 min
    # recompile each time). Mount Google Drive first, then:
    import os
    os.environ["VLLM_XLA_CACHE_PATH"] = "/content/drive/MyDrive/xla_cache"

    # Verify installation
    python -c "import jax; import vllm; print(jax.devices())"

Usage:
    # Standard AccelMark run
    python run.py --runner google_vllm_tpu_{hash8} --suite suite_F

    # Override model path if downloaded locally
    python run.py --runner google_vllm_tpu_{hash8} --suite suite_F \\
        --model-path /path/to/Qwen2.5-0.5B-Instruct

    # Skip XLA precompilation step (for quick testing, not for benchmarking)
    SKIP_JAX_PRECOMPILE=1 python run.py --runner google_vllm_tpu_{hash8} \\
        --suite suite_F --scenario accuracy
"""

import gc
import os
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
logging.getLogger("vllm.engine.llm_engine").setLevel(logging.WARNING)

# ── Colab / Jupyter compatibility — apply at import time ─────────────────────
# These patches must be in place before LLM() is called, regardless of whether
# the runner is invoked from a notebook cell or a subprocess.

# Fix 1: disable vLLM subprocess engine spawning.
# The spawned process calls sys.stdout.fileno() which crashes in Jupyter because
# ipykernel wraps stdout without fileno() support. In-process mode is
# functionally identical for single-chip TPU.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# Fix 2: patch suppress_stdout() to be a no-op when fileno() is unavailable.
# vllm calls this during GroupCoordinator.__init__ even in in-process mode.
# We patch both the source module and the already-imported reference in
# parallel_state, since Python's import system gives each module its own
# reference to the function.
try:
    from contextlib import contextmanager as _cm
    import vllm.utils.system_utils as _su
    import vllm.distributed.parallel_state as _ps

    _orig_suppress = _su.suppress_stdout

    @_cm
    def _safe_suppress_stdout():
        try:
            sys.stdout.fileno()       # raises in Jupyter, works in terminal
            with _orig_suppress():
                yield
        except Exception:
            yield                     # Jupyter: skip suppression silently

    _su.suppress_stdout = _safe_suppress_stdout
    _ps.suppress_stdout = _safe_suppress_stdout
except Exception:
    pass  # vllm not installed yet — patches will be applied on first import
# ─────────────────────────────────────────────────────────────────────────────


class GoogleTPUVLLMRunner(BenchmarkRunner):
    """
    AccelMark benchmark runner using vllm-tpu on Google TPUs.

    vllm-tpu uses the standard vLLM LLM constructor but compiles models via
    JAX/XLA. The inference API is synchronous (generate), not streaming.
    All TPU-specific behaviour is isolated to load_model(), release_resources(),
    and the capability flags below.
    """

    # ── Capability flags ──────────────────────────────────────────────────────

    SUPPORTS_STREAMING = False
    """
    vllm-tpu's platform layer returns is_async_output_supported = False,
    meaning token-by-token streaming is not available. Online, interactive,
    and sustained scenarios require streaming and will be skipped.
    Only offline and accuracy scenarios run on this runner.
    """

    SUPPORTS_BATCHING = True
    """
    TPU LLM.generate() accepts batches of prompts — fully supported.
    """

    SUPPORTS_ONLINE = False
    """
    Online scenario requires streaming; skipped for this runner.
    """

    SUPPORTS_MULTI_CHIP = False
    """
    v5e-1 Colab is a single-chip setup. Multi-chip tensor parallelism
    requires a v5e-4 or larger slice. Set False so the framework ignores
    any --tensor-parallel-size flag and always uses 1 chip.
    """

    SUPPORTED_PRECISIONS = ["bf16"]
    """
    v5e natively supports BF16 for LLM inference. FP16 and FP32 are not
    first-class dtypes on this generation. BenchmarkRunner will auto-select
    BF16 and warn if a suite requires FP16.
    """

    SUPPORTED_QUANTIZATION_BACKENDS = []
    """
    Quantization on v5e is hardware-generation dependent:
      - INT8 W8A8 / W4A16 AWQ listed as v5/v6-capable in tpu-inference
        support matrices, but marked "Untested" as of the release matrix.
      - FP8 recommended for v7x only.
    Leave empty for safety — Suite C will be skipped on this runner.
    Revisit when tpu-inference validates INT8 on v5e.
    """

    # ── Initialiser ───────────────────────────────────────────────────────────

    def __init__(self):
        self.llm             = None   # vllm.LLM instance
        self.tokenizer       = None   # HuggingFace tokenizer
        self.sampling_params = None   # vllm.SamplingParams

    # ── Platform helpers ──────────────────────────────────────────────────────

    def _get_chip_count(self) -> int:
        """
        Count available TPU chips via /dev/accel* device files.
        tpu_info.get_num_chips() counts these files — returns 1 on v5e-1 Colab.
        Falls back to JAX device count, then 1.
        """
        try:
            from tpu_inference import tpu_info
            n = tpu_info.get_num_chips()
            if n > 0:
                return n
        except Exception:
            pass
        try:
            import jax
            n = len(jax.devices())
            if n > 0:
                return n
        except Exception:
            pass
        return 1

    def _get_framework_name(self) -> str:
        return "vllm-tpu"

    def _get_framework_version(self) -> str:
        """
        Report tpu-inference plugin version — this is the TPU-specific package
        and the meaningful version for reproducibility on this hardware.
        Falls back to vllm core version.
        """
        try:
            from importlib.metadata import version
            return version("tpu_inference")
        except Exception:
            pass
        try:
            from importlib.metadata import version
            return version("vllm-tpu")
        except Exception:
            pass
        try:
            import vllm
            return vllm.__version__
        except Exception:
            return "unknown"

    def get_peak_memory_gb(self) -> Optional[float]:
        """
        TPU HBM memory is not exposed via a PyTorch-style max_memory_allocated()
        API. Returns None — memory reporting is not available on this platform.
        """
        return None

    def get_supported_precisions(self, chip_name: str, env_info: dict):
        """
        TPU v5e supports BF16 natively. Override hardware detection to always
        return BF16 — the auto-detector cannot read TPU compute capability
        and would default to BF16+FP16+FP32, which is misleading here.
        """
        return ["BF16"]

    def get_effective_dtype(self) -> Optional[str]:
        return getattr(self, "_effective_dtype", "bfloat16")

    def get_model_format(self) -> str:
        return "HuggingFace original"

    # ── Required: load model ──────────────────────────────────────────────────

    def load_model(self, model_path: str, parallelism: dict) -> None:
        """
        Load model onto TPU via vllm-tpu (LLM constructor).

        vllm-tpu activates when the tpu-inference plugin is installed and
        JAX_PLATFORMS="tpu" (set automatically by the plugin). No explicit
        device= kwarg is needed — the platform is detected at import time.

        XLA compilation: first call to generate() triggers XLA graph compilation
        per input shape bucket. This takes 20–30 minutes on first run, ~5 min
        on subsequent runs (cache hit). The benchmark warmup_runs in AccelMark
        LoadGen handle this — the first run is discarded and is where compile
        time is absorbed.

        On Colab, persist the XLA cache to Google Drive:
            os.environ["VLLM_XLA_CACHE_PATH"] = "/content/drive/MyDrive/xla_cache"

        Args:
            model_path:  Resolved HuggingFace model ID or local path.
            parallelism: Engine config dict from BenchmarkRunner. On v5e-1,
                         tensor_parallel_size is always 1 (SUPPORTS_MULTI_CHIP
                         = False), use_async is always False (SUPPORTS_STREAMING
                         = False).
        """
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        max_tokens    = parallelism["max_tokens"]
        max_model_len = parallelism.get("max_model_len")

        cfg             = getattr(self, "_runner_config", {})
        max_num_seqs    = cfg.get("max_num_seqs", 512)   # vLLM default; override in runner config YAML
        gpu_memory_util = cfg.get("gpu_memory_utilization", 0.90)
        extra_kwargs    = dict(cfg.get("engine_kwargs") or {})

        # v5e-1 has only 16 GiB HBM. For 8B models, model weights alone
        # consume ~16 GB in BF16, leaving almost no KV cache headroom.
        # max_model_len defaults conservatively if not set by the suite.
        if max_model_len is None:
            max_model_len = 2048
            print(f"  Note: max_model_len not set by suite — defaulting to "
                  f"{max_model_len} to fit within 16 GiB TPU HBM.")

        print(f"Loading model on TPU: dtype=bfloat16, "
              f"max_model_len={max_model_len}, max_num_seqs={max_num_seqs}")
        print("  Note: First run triggers XLA graph compilation (~20-30 min).")
        print("  Set VLLM_XLA_CACHE_PATH to a persistent path to cache across sessions.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )

        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )

        llm_kwargs = dict(
            model=model_path,
            dtype="bfloat16",
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            # gpu_memory_utilization controls KV cache size on TPU —
            # same parameter name, TPU-specific interpretation.
            gpu_memory_utilization=gpu_memory_util,
            trust_remote_code=False,
            **extra_kwargs,
        )

        self.llm = LLM(**llm_kwargs)
        self._effective_dtype = "bfloat16"

    # ── Required: offline batch inference ────────────────────────────────────

    def inference_fn_offline(
        self, requests: list[InferenceRequest]
    ) -> list[InferenceResult]:
        """
        Synchronous batch inference via vllm-tpu LLM.generate().

        All prompts are submitted as a batch. XLA compiles a graph per unique
        (input_len_bucket, output_len) shape on first encounter, caching results
        for subsequent runs.

        Args:
            requests: List of InferenceRequest objects.

        Returns:
            List[InferenceResult], same length and order as requests.
        """
        prompts = [self._format_prompt(r.prompt) for r in requests]

        # Build per-request SamplingParams if max_tokens varies per request.
        # Most suites use a uniform max_tokens set at load_model() time.
        sampling = self.sampling_params

        t_start = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        results = []
        for output in outputs:
            out = output.outputs[0]
            results.append(InferenceResult(
                first_token_time_ms=None,   # streaming not available on TPU
                total_time_ms=elapsed_ms,   # wall-clock for entire batch
                output_tokens=len(out.token_ids),
                input_tokens=len(output.prompt_token_ids) if output.prompt_token_ids else 0,
                success=True,
                output_text=out.text,
            ))
        return results

    # ── Required: resource cleanup ────────────────────────────────────────────

    def release_resources(self) -> None:
        """
        Release TPU HBM and Python references.

        vllm-tpu does not expose an explicit engine teardown method. Deleting
        the LLM reference and running gc.collect() releases Python-side
        references. The XLA-compiled graph cache persists on disk — this is
        intentional for fast subsequent runs.
        """
        if self.llm is not None:
            del self.llm
            self.llm = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.sampling_params is not None:
            del self.sampling_params
            self.sampling_params = None
        gc.collect()

        # Attempt to clear JAX device memory (best-effort).
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass

    # ── Optional overrides ────────────────────────────────────────────────────

    def format_prompt(self, prompt: str) -> str:
        """Apply chat template if tokenizer has one."""
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def get_extra_subprocess_args(self, args) -> list[str]:
        """No custom CLI flags to forward to scenario subprocesses."""
        return []


if __name__ == "__main__":
    GoogleTPUVLLMRunner().main()