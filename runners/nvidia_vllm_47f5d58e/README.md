# nvidia_vllm_47f5d58e — NVIDIA vLLM Runner

AccelMark runner for NVIDIA GPUs using [vLLM](https://github.com/vllm-project/vllm).

## Supported suites

| Suite | Description | Notes |
|-------|-------------|-------|
| Suite A | Single-chip, Llama-3-8B | Reference suite; supports speculative and burst extra scenarios |
| Suite B | Multi-chip, Llama-3-70B | Requires 4× A100/H100 or equivalent; supports burst extra scenario |
| Suite C | Quantization, Llama-3.1-8B | FP8 requires Hopper (H100); W8A8/W4A16 on Ampere+ |
| Suite D | Long context ~28K input, Llama-3.1-8B (`max_model_len` 30,208) | Supports speculative extra scenario |
| Suite E | Multi-chip scaling, Llama-3-8B | Requires NVLink for meaningful scaling results |
| Suite F | Consumer/edge, Qwen2.5-0.5B | Single-chip; pre-Ampere supported with `--enforce-eager` |
| Suite G | MoE multi-chip, Mixtral-8x7B-Instruct-v0.1 | Requires ≥2× A100-80GB (or ≥4× A100-40GB); `required_chips: auto` uses all available GPUs |

## Hardware compatibility

| GPU generation | Architecture | sm | BF16 | CUDA graphs | Suite F | Notes |
|---|---|---|---|---|---|---|
| H100, H800 | Hopper | sm_90 | ✅ | ✅ | ✅ | FP8 native; full support |
| H20 | Hopper | sm_90 | ✅ | ✅ | ✅ | See large-memory notes below |
| A100, A800, A10, L4, L40 | Ampere | sm_80/86 | ✅ | ✅ | ✅ | Full support |
| RTX 3090, RTX 3080, A5000 | Ampere | sm_86 | ✅ | ✅ | ✅ | Full support |
| RTX 4090, RTX 4080, RTX 4070 | Ada Lovelace | sm_89 | ✅ | ✅ | ✅ | Full support |
| RTX 2080, T4 | Turing | sm_75 | ❌ | ⚠️ | ✅ | See pre-Ampere notes below |
| V100, V100S | Volta | sm_70 | ❌ | ⚠️ | ✅ | See pre-Ampere notes below |

## Large-memory GPUs (H20, A100 80 GB, etc.)

On GPUs with very large VRAM (≥ 80 GB per chip), vLLM allocates a proportionally large
KV cache. Combined with an outdated `nvidia-cublas-cu12`, this can trigger a **SIGFPE**
(floating point exception) during inference — the process exits silently with no Python
traceback.

**Symptom:** accuracy gate subprocess exits with `SIGFPE (return code -8)`, either
immediately after model load or at the first inference batch.

**Fix: upgrade cuBLAS:**

```bash
pip install --upgrade nvidia-cublas-cu12
```

This is a bug in older cuBLAS builds that manifests specifically on large-memory GPUs
where the KV cache allocation is large enough to trigger a code path with an uninitialized
state. Upgrading cuBLAS resolves it without any configuration changes.

## Pre-Ampere GPUs (V100, T4, RTX 20xx)

Pre-Ampere GPUs (sm < 80) have two known issues with recent vLLM versions:

**1. No native BF16**

These GPUs do not support BF16 compute. The runner detects this automatically and
falls back to FP16 via `_resolve_precision()`. Suite F declares `allowed_precisions:
[BF16, FP16]`, so the fallback to FP16 on pre-Ampere hardware is silent — no warning,
since FP16 is an explicitly accepted precision. Results are labeled with the actual
precision used (FP16 on pre-Ampere, BF16 on Ampere+). Suites A–F also allow FP16
as a fallback and will warn when it is used.

**2. Triton `sm >= 80` assertion (CUDA graphs)**

vLLM's default attention kernels use Triton flash-attention which asserts Ampere or
newer at runtime. On Volta/Turing this causes a crash or OOM during CUDA graph capture.

**Fix: add `--enforce-eager` to disable CUDA graph capture:**

```bash
# Suite F on V100 or T4
python runners/nvidia_vllm_47f5d58e/runner.py \
    --suite suite_F \
    --enforce-eager
# Or set persistently in configs/runner_configs/runner_nvidia_vllm_47f5d58e.yaml:
# enforce_eager: true  (under global defaults or suites.suite_F)

# Suite A on V100 (1× only — Suite E 2×/4× remain blocked on V100)
python runners/nvidia_vllm_47f5d58e/runner.py \
    --suite suite_A \
    --enforce-eager
# Or set persistently in configs/runner_configs/runner_nvidia_vllm_47f5d58e.yaml:
# enforce_eager: true  (under global defaults or suites.suite_A)
```

`--enforce-eager` disables CUDA graph capture entirely and falls back to eager PyTorch
execution. This has a throughput cost (typically 10–20% on Ampere, less significant
on Volta where CUDA graphs provide less benefit anyway).

**Suite E on V100**

- 1× passes with `--enforce-eager`
- 2× fails with `returncode=-9` (SIGKILL/OOM) even with `--enforce-eager` — the CUDA
  graph memory profiler overestimates available headroom at TP=2 on 32 GB V100
- 4× fails with `returncode=1` (Triton `sm >= 80` assertion regardless of eager mode)

Recommended: run Suite E on V100 with `--enforce-eager` (or set `enforce_eager: true`
in the runner config yaml). Limit chip count via `tensor_parallel_size: 1` in the
runner config or by passing `--tensor-parallel-size 1` on the CLI.
Use Suite F
for single-chip consumer benchmarking on V100.

## Basic usage

```bash
# Run Suite F (consumer/edge benchmark)
python runners/nvidia_vllm_47f5d58e/runner.py --suite suite_F

# Run Suite A (standard datacenter benchmark)
python runners/nvidia_vllm_47f5d58e/runner.py --suite suite_A

# Run a single scenario
python runners/nvidia_vllm_47f5d58e/runner.py --suite suite_F --scenario offline

# Use a local model cache
python runners/nvidia_vllm_47f5d58e/runner.py --suite suite_F \
    --model-path /data/models/Qwen2.5-0.5B-Instruct

# Pre-Ampere GPU (V100, T4, RTX 20xx)
python runners/nvidia_vllm_47f5d58e/runner.py --suite suite_F --enforce-eager
# Or set persistently in configs/runner_configs/runner_nvidia_vllm_47f5d58e.yaml:
# enforce_eager: true  (under global defaults or suites.suite_F)
```

## Requirements

See `requirements.txt` for the pinned dependency list.

Minimum environment:
- NVIDIA GPU with compute capability ≥ 7.0 (Volta or newer)
- CUDA 11.8+ (12.x recommended)
- Python 3.10+
- vLLM ≥ 0.4.0 (≥ 0.6.0 recommended; CUDA 11.8 users are limited to ≤ 0.5.5)

## TODO

- **`get_runtime_metrics()` override for speculative decoding.** Add an override in `runner.py` to expose speculative decoding acceptance rate. Check vLLM's `AsyncEngineClient.get_decoding_stats()` or equivalent for the correct API call across vLLM 0.4.0–0.7.x. This triggers a runner hash change and `supersedes_chain` update. Prerequisite: test end-to-end speculative scenario run on hardware.
- **SGLang equivalent.** Add the same `get_runtime_metrics()` override in the SGLang runner once the SGLang speculative stats API is confirmed.
