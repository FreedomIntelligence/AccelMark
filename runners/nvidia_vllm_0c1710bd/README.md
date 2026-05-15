# nvidia_vllm_0c1710bd — NVIDIA vLLM Runner (0.20.x line)

AccelMark reference runner for NVIDIA GPUs running **vLLM 0.20.x** —
the 2026 major release.

This runner supersedes [`nvidia_vllm_47f5d58e`](../nvidia_vllm_47f5d58e/)
(vLLM 0.7.3). The predecessor remains runnable; this folder is what new
results on Ampere / Hopper / Blackwell hosts should reference going forward.

## What changed vs nvidia_vllm_47f5d58e

| Area | 0.7.3 (predecessor) | 0.20.x (this runner) |
|---|---|---|
| Default CUDA | 12.1 | **13.0** (12.8 still supported via the PyTorch cu128 index) |
| PyTorch | 2.5.1 | **2.11.0** |
| Python | 3.10+ | 3.10+ (3.14 newly supported) |
| HuggingFace Transformers | v4.57 | **v5.x** |
| FlashAttention | FA2 | **FA4** (MLA prefill default) |
| Quantization backends declared | fp8, compressed-tensors, gptq_marlin | + **turboquant** (2-bit KV cache, 4x KV capacity) |
| Model Runner | V1 | **V2** (Eagle prefill full-CUDA-graph, fused probabilistic rejection sampling) |
| DeepSeek V4 | — | ✅ |
| Result version string | `vllm 0.7.3` | `vllm 0.20.1+transformers-5.1.0` |

Detailed release notes:
[vLLM v0.20.0](https://github.com/vllm-project/vllm/releases/tag/v0.20.0)
· [vLLM v0.20.1 patch](https://github.com/vllm-project/vllm/releases/tag/v0.20.1).

## Supported suites

Same coverage as the predecessor runner — **all suites A–G**. See
[`runners/nvidia_vllm_47f5d58e/README.md`](../nvidia_vllm_47f5d58e/README.md)
for the per-GPU hardware compatibility matrix; the same rows apply here
because the runner code is a structural clone.

## Installation

```bash
# 1. Standard install — CUDA 13.0 stack
pip install -r runners/nvidia_vllm_0c1710bd/requirements.txt

# 2. CUDA 12.8 stack (for hosts still on the cu128 driver):
pip install -r runners/nvidia_vllm_0c1710bd/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

> Older runners pinned `nvidia-cublas-cu12`; on 0.20 + CUDA 13.0 use
> `nvidia-cublas-cu13` if you encounter the cuBLAS SIGFPE on large-memory
> GPUs (same fix philosophy as the predecessor's README — only the package
> name changes).

## Basic usage

Identical to the predecessor:

```bash
python run.py --runner nvidia_vllm_0c1710bd --suite suite_A
python run.py --runner nvidia_vllm_0c1710bd --suite suite_B \
    --tensor-parallel-size 4
```

## 0.20-specific knobs you may want to enable

`engine_kwargs` in the runner config are passed straight to
`AsyncEngineArgs` / `LLM`. The runner already filters unknown fields, so
adding 0.20-only keys is safe even if you downgrade vLLM later — they will
be dropped with a warning rather than blowing up at startup.

```yaml
# configs/runner_configs/runner_nvidia_vllm_0c1710bd.yaml
engine_kwargs:
  # FlashAttention 4 (default on 0.20 — listed here only if you need
  # to pin it for reproducibility):
  attention_backend: FLASH_ATTN_4
  # CUDA graph improvements added in 0.20:
  compilation_config:
    cudagraph_mode: full_and_piecewise
  # TurboQuant 2-bit KV cache (suite C with --precision turboquant):
  # kv_cache_dtype: turboquant
```

## Runner config

Copy the example:

```bash
cp configs/runner_configs/runner_nvidia_vllm_0c1710bd.yaml.example \
   configs/runner_configs/runner_nvidia_vllm_0c1710bd.yaml
```

Field names and defaults are identical to the predecessor — see
[`runner_nvidia_vllm_47f5d58e.yaml.example`](../../configs/runner_configs/runner_nvidia_vllm_47f5d58e.yaml.example)
for the field reference.

## Status

- **Code:** structurally identical to the predecessor + the small additions
  documented above. The change is principally a dependency bump.
- **Validation:** not yet run end-to-end on a 0.20 install at the time of
  commit. The predecessor's test_smoke.py path applies once the test file is
  ported over.
- **Predecessor:** `nvidia_vllm_47f5d58e/meta.json` will receive a
  `deprecated_by` pointer in a follow-up PR once a smoke result against this
  runner has been verified. Until then the predecessor remains the
  recommended runner for production result submissions.
