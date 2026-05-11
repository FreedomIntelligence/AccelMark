"""
Fast smoke test for vllm-tpu on Google TPU v5e-1.

Model: Qwen/Qwen3-0.6B  (Qwen3ForCausalLM — natively supported by tpu-inference)

NOTE on model choice:
    Qwen2.5-0.5B-Instruct (Qwen2ForCausalLM) is NOT supported by tpu-inference's
    JAX-native model registry. It falls back to PyTorch/torchax and hits a
    recursive JAX JIT error during XLA compilation. Use Qwen3 or Llama3 instead:

    Supported architectures (tpu-inference JAX-native):
      Qwen3ForCausalLM   → Qwen/Qwen3-0.6B, Qwen/Qwen3-4B
      LlamaForCausalLM   → meta-llama/Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct

Run from the AccelMark repo root:
    python runners/google_vllm_tpu_68cc9ffa/test_smoke.py

Optional — persist XLA cache to Google Drive (Colab) to avoid
recompiling XLA graphs on every session restart:
    VLLM_XLA_CACHE_PATH=/content/drive/MyDrive/vllm_xla_cache \\
    python runners/google_vllm_tpu_68cc9ffa/test_smoke.py
"""

import gc
import os
import sys
import time
from contextlib import contextmanager

# ── Colab / Jupyter compatibility fixes ───────────────────────────────────────
# Must happen before any vllm import.

# Fix 1: disable subprocess engine spawning. vLLM's EngineCore subprocess calls
# sys.stdout.fileno() which crashes in Jupyter (ipykernel wraps stdout without
# fileno support). In-process mode is functionally identical for single-chip TPU.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# Fix 2: patch suppress_stdout() in vllm to be a no-op when fileno() is
# unavailable. Even in in-process mode, vllm calls suppress_stdout() during
# distributed init (GroupCoordinator.__init__). Jupyter's stdout raises
# UnsupportedOperation on fileno(), so we make the suppression optional.
import vllm.utils.system_utils as _su
import vllm.distributed.parallel_state as _ps

_original_suppress_stdout = _su.suppress_stdout

@contextmanager
def _safe_suppress_stdout():
    try:
        sys.stdout.fileno()           # raises in Jupyter, works in terminal
        with _original_suppress_stdout():
            yield
    except Exception:
        yield                         # Jupyter: skip suppression, carry on

_su.suppress_stdout = _safe_suppress_stdout
_ps.suppress_stdout = _safe_suppress_stdout
# ─────────────────────────────────────────────────────────────────────────────

from vllm import LLM, SamplingParams

MODEL = os.getenv("MODEL_PATH", "Qwen/Qwen3-0.6B")

prompts = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

print(f"Loading {MODEL} ...")
print("Note: first run compiles XLA graphs — may take 20-30 min.")
t_load = time.perf_counter()
llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=1024,
    max_num_seqs=4,
    gpu_memory_utilization=0.95,
)
print(f"Model loaded in {time.perf_counter() - t_load:.1f}s\n")

t_infer = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
print(f"Inference done in {time.perf_counter() - t_infer:.1f}s\n")

for output in outputs:
    print(f"Prompt:  {output.prompt!r}")
    print(f"Output:  {output.outputs[0].text!r}")
    print()

del llm
gc.collect()
print("Resources released. Done.")
