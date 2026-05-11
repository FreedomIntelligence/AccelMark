# google_vllm_tpu_68cc9ffa — Google TPU Runner (vllm-tpu)

AccelMark runner for Google Cloud TPUs using [vllm-tpu](https://github.com/vllm-project/tpu-inference),
the JAX/XLA backend for vLLM developed by Google. Uses the standard vLLM Python API
(`LLM`, `SamplingParams`) but compiles models with XLA rather than CUDA graphs.

## Supported suites

| Suite | Description | Status |
|-------|-------------|--------|
| Suite A | Single-chip, Llama-3-8B | ✅ Verified on v6e-1 (offline + speculative scenarios) |
| Suite B | Multi-chip, Llama-3-70B | ❌ Skipped — `SUPPORTS_MULTI_CHIP = False` |
| Suite C | Quantization, Llama-3.1-8B | ❌ Skipped — `SUPPORTED_QUANTIZATION_BACKENDS = []` (INT8/W4A16 untested on v5e/v6e, FP8 v7x-only) |
| Suite D | Long context ~28K input, Llama-3.1-8B | ✅ Verified on v6e-1 (offline scenario; ~23 min runtime) |
| Suite E | Multi-chip scaling, Llama-3-8B | ❌ Skipped — `SUPPORTS_MULTI_CHIP = False` |
| Suite F | Consumer/edge, Qwen2.5-0.5B | ✅ Verified on v5e-1 and v6e-1 (requires Qwen3-0.6B substitution via `models_local.yaml`; see "Model support") |
| Suite G | MoE multi-chip, Mixtral-8x7B | ❌ Skipped — `SUPPORTS_MULTI_CHIP = False` |

Online / interactive / sustained scenarios are also skipped:
`SUPPORTS_STREAMING = False` because vllm-tpu's platform layer returns
`is_async_output_supported = False`. Only **offline**, **speculative**, and **accuracy**
scenarios run on this runner.

## Hardware compatibility

| TPU | HBM | BF16 | Multi-chip TP | FP8 | Quantization | Status |
|-----|-----|------|---------------|-----|--------------|--------|
| v5e-1 (Colab) | 16 GiB | ✅ | ❌ (single chip) | ❌ | ❌ | ✅ Verified — Suite F only (Llama-3-8B is too tight for 16 GiB) |
| v6e-1 | 32 GiB | ✅ | ❌ (single chip) | ❌ | ❌ | ✅ Verified — Suites A, D, F |
| v5e-4 / v5e-8 | 16 GiB/chip | ✅ | ⚠️ (set `SUPPORTS_MULTI_CHIP = True`) | ❌ | Untested | Should work without code changes |
| v6e-4 / v6e-8 | 32 GiB/chip | ✅ | ⚠️ (untested) | ❌ | Untested | Should work without code changes |
| v7x | (varies) | ✅ | ⚠️ (untested) | ⚠️ (recommended for v7x only, untested) | Untested | Should work without code changes |

BF16 is the only supported precision on this runner — FP16 and FP32 are not
first-class dtypes for LLM inference on TPU. `get_supported_precisions()` always
returns `["BF16"]`.

## Model support

`tpu-inference` compiles models via JAX/XLA from a native model registry. Models not in
the registry fall back to PyTorch/`torchax`, which hits a recursive JIT error on v5e/v6e.

**Supported (JAX-native, confirmed working):**

| Architecture | Example models |
|---|---|
| `LlamaForCausalLM` | `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Meta-Llama-3-8B-Instruct` |
| `Qwen3ForCausalLM` | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-32B` |

**NOT supported (falls back to PyTorch, crashes with `RecursionError`):**

| Architecture | Example models |
|---|---|
| `Qwen2ForCausalLM` | `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` |

Suite F's default model `Qwen2.5-0.5B-Instruct` (Qwen2ForCausalLM) will not run until
`tpu-inference` adds Qwen2 to its JAX registry. Add this entry to
`configs/models_local.yaml` to substitute Qwen3-0.6B (the verified runs above use
this substitution):

```yaml
models:
  Qwen/Qwen2.5-0.5B-Instruct:
    local_path: Qwen/Qwen3-0.6B
    model_name: Qwen/Qwen3-0.6B
    note: "Qwen2ForCausalLM not supported by tpu-inference; using Qwen3-0.6B."
```

## Prerequisites

**Option A — PyPI (recommended, simplest):**

```bash
pip install vllm-tpu
pip install -r runners/google_vllm_tpu_68cc9ffa/requirements.txt
```

`vllm-tpu` is self-contained — it pulls in the correct vLLM build, JAX, `libtpu`, and
the `tpu-inference` plugin in one step. Do **not** separately install `torch`,
`torch-xla`, or `jax` — `vllm-tpu` manages these versions internally.

The verified runs above used `vllm-tpu 0.13.3` (which bundles JAX 0.8.1 +
`tpu_inference 0.13.3`) — pin to this version for reproducibility.

**Option B — Docker (TPU VM / GKE):**

```bash
docker pull vllm/vllm-tpu:latest
docker run --privileged --net=host --shm-size=150gb -it vllm/vllm-tpu:latest
# Inside the container:
pip install -r runners/google_vllm_tpu_68cc9ffa/requirements.txt
```

**Option C — Source (advanced, for debugging vllm-tpu internals):**

See the header of `requirements.txt` for the exact `VLLM_COMMIT` and source-build steps.

**Verify installation:**

```bash
python -c "
import jax, vllm
from importlib.metadata import version
print('vllm          :', vllm.__version__)
print('tpu_inference :', version('tpu_inference'))
print('jax           :', jax.__version__)
print('jax devices   :', jax.devices())
"
# Expected on Colab v5e-1:
# jax devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
```

## Basic usage

```bash
# Run Suite F (consumer/edge benchmark — needs Qwen3-0.6B substitution)
python run.py --runner google_vllm_tpu_68cc9ffa --suite suite_F

# Run Suite A (standard datacenter benchmark — verified on v6e-1)
python run.py --runner google_vllm_tpu_68cc9ffa --suite suite_A

# Run Suite D (long-context 28K input — verified on v6e-1, ~23 min)
python run.py --runner google_vllm_tpu_68cc9ffa --suite suite_D

# Run a single scenario
python run.py --runner google_vllm_tpu_68cc9ffa --suite suite_F --scenario offline

# Use a local model cache
python run.py --runner google_vllm_tpu_68cc9ffa --suite suite_F \
    --model-path /path/to/Qwen3-0.6B

# Skip XLA precompilation step (quick test only, not for benchmarking)
SKIP_JAX_PRECOMPILE=1 python run.py --runner google_vllm_tpu_68cc9ffa \
    --suite suite_F --scenario accuracy
```

## Runner config

Copy the example config and adjust for your TPU slice:

```bash
cp configs/runner_configs/runner_google_vllm_tpu_68cc9ffa.yaml.example \
   configs/runner_configs/runner_google_vllm_tpu_68cc9ffa.yaml
```

The example config is **tuned conservatively for v5e-1 (16 GiB HBM)**. On v6e-1 (32 GiB HBM)
the verified runs used the framework defaults (`max_num_seqs: 512`, `gpu_memory_utilization: 0.9`)
without per-suite overrides — feel free to delete the suite overrides if you are on v6e-1+.

Key settings:

| Field | v5e-1 default (example) | v6e-1 verified | Notes |
|-------|------------------------|----------------|-------|
| `max_num_seqs` (global) | 32 | 512 | vLLM's own default is 256; on v5e-1 lower because HBM is constrained |
| `gpu_memory_utilization` (global) | 0.90 | 0.90 | Fraction of TPU HBM reserved for KV cache; reduce if OOM |
| `suites.suite_A.max_num_seqs` | 1 | 512 | Llama-3-8B in BF16 fills nearly all 16 GiB on v5e-1; 32 GiB v6e-1 has plenty of headroom |
| `suites.suite_A.gpu_memory_utilization` | 0.88 | 0.90 | |
| `suites.suite_F.max_num_seqs` | 32 | 512 | Qwen3-0.6B is tiny — high concurrency is fine |
| `suites.suite_F.gpu_memory_utilization` | 0.95 | 0.90/0.95 | v5e-1 verified at 0.95, v6e-1 at 0.90 |
| `suites.suite_D.max_num_seqs` | 1 | 512 | Long-context (~28K input); v5e-1 will OOM, v6e-1 has the headroom for default settings |
| `suites.suite_D.gpu_memory_utilization` | 0.85 | 0.90 | |

## XLA compilation cache

The first inference run compiles XLA graphs for each input shape bucket, taking
**20–30 minutes** for Llama-class models. Subsequent runs with a warm cache take
~5 minutes for the same shape buckets.

On Colab (ephemeral sessions), persist the cache to Google Drive so you don't
recompile on every session restart. Add this to your notebook **before** running
any AccelMark commands:

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["VLLM_XLA_CACHE_PATH"] = "/content/drive/MyDrive/vllm_xla_cache"
```

On TPU VMs, point `VLLM_XLA_CACHE_PATH` at any persistent local path.

## TPU OOM errors

v5e-1 has only 16 GiB HBM per chip. Llama-3-8B in BF16 uses ~16 GB just for weights,
leaving almost no KV cache headroom — that is why the example config caps Suite A
and Suite D at `max_num_seqs: 1` on v5e-1. Switch to v6e-1 (32 GiB) for those suites.

If you still see OOM, reduce `gpu_memory_utilization` and/or `max_num_seqs` in the
runner config YAML, either globally or per-suite:

```yaml
gpu_memory_utilization: 0.85
max_num_seqs: 1

suites:
  suite_D:
    max_num_seqs: 1
    gpu_memory_utilization: 0.80
```

Also consider lowering the suite-level `max_model_len` if your prompts are short —
the runner defaults to `2048` when a suite does not set one (see `load_model()`).

## Colab / Jupyter compatibility

The runner applies two compatibility patches at import time so it works inside
notebook kernels:

1. `VLLM_ENABLE_V1_MULTIPROCESSING=0` — disables vLLM's subprocess engine, which
   would otherwise call `sys.stdout.fileno()` and crash under ipykernel.
2. `vllm.utils.system_utils.suppress_stdout` is monkey-patched to a no-op when
   `fileno()` is unavailable. This is also patched on the
   `vllm.distributed.parallel_state` module-level reference.

In-process mode is functionally identical for single-chip TPU — there is no
performance impact from these patches.

## Requirements

See `requirements.txt` for the pinned dependency list.

Minimum environment:
- Google Cloud TPU v5e, v6e, or v7x
- Python 3.10+ (verified on 3.12.13)
- `vllm-tpu` 0.13.3 (bundles vLLM, JAX 0.8.1, `libtpu`, and `tpu-inference 0.13.3`)
- `tpu-info` ≥ 0.7.1 (for `/dev/accel*` chip detection)

## Limitations / TODO

- **Multi-chip TP.** Set `SUPPORTS_MULTI_CHIP = True` and validate on a v5e-4 or
  v6e-4 slice. Current setting hard-codes `tensor_parallel_size = 1`.
- **Quantization.** INT8 W8A8 and W4A16 AWQ are listed as v5/v6-capable in
  `tpu-inference` support matrices but marked "Untested". Add to
  `SUPPORTED_QUANTIZATION_BACKENDS` once validated on hardware. FP8 is v7x-only.
- **Streaming.** `is_async_output_supported = False` in the vllm-tpu platform layer.
  Online / interactive / sustained scenarios will run once Google ships an
  `AsyncLLMEngine` for tpu-inference.
- **Memory reporting.** `get_peak_memory_gb()` returns `None` — TPU HBM is not
  exposed via a PyTorch-style `max_memory_allocated()` API. Investigate `tpu-info`
  or `libtpu` HBM usage hooks for an equivalent metric.
- **Speculative decoding metrics.** vllm-tpu does not surface acceptance-rate
  stats, so Suite A's speculative scenario currently reports the same throughput
  as offline. Add a `get_runtime_metrics()` override once Google exposes these.
- **Qwen2 / additional architectures.** Suite F's default Qwen2.5-0.5B does not run
  until `tpu-inference` adds `Qwen2ForCausalLM` to its JAX model registry.
