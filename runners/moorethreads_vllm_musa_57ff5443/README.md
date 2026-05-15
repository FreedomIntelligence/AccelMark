# moorethreads_vllm_musa_57ff5443 — Moore Threads MUSA Runner (vllm-musa)

AccelMark runner for Moore Threads MUSA GPUs using
[vllm-musa](https://github.com/MooreThreads/vllm-musa), the official vLLM
platform plugin for MUSA hardware.

> **Status:** This runner is **untested on real silicon at the time of
> commit**. The code is written against the public `vllm-musa` plugin
> documentation and follows the structural template of the
> `ascend_vllm_ascend_*` runner. Plan to smoke-test on an S5000 / S4000
> system; capability flags and dtype mappings may be adjusted in a follow-up
> runner version (new hash, new folder) based on real-world findings.

## How vllm-musa works

`vllm-musa` is a vLLM **platform plugin** (auto-detected on `import vllm`)
that makes the standard vLLM Python API run on Moore Threads MUSA GPUs. It
relies on three components:

| Component | Role |
|---|---|
| `torchada` | CUDA→MUSA compatibility layer for PyTorch — aliases `torch.cuda.*` to MUSA so most code paths run unmodified |
| `pymtml` (`mthreads-ml-py`) | Moore Threads Management Library bindings, equivalent to `nvidia-ml-py` |
| Triton patches | Runtime monkey-patches in `vllm_musa_platform.patches.*` that fix `triton.attention` and `worker` modules for MUSA's Triton compiler |

The standard `vllm.LLM`, `vllm.AsyncLLMEngine`, and `vllm.SamplingParams`
remain the entry points — this runner therefore reuses ~95% of the logic
from the NVIDIA / Ascend vLLM runners.

## Supported suites

| Suite | Description | Notes |
|-------|-------------|-------|
| Suite A | Single-chip, Llama-3-8B | Pending smoke test on S4000 / S5000 |
| Suite B | Multi-chip, Llama-3-70B | Requires multiple Moore Threads cards + MCCL TP |
| Suite C | Quantization, Llama-3.1-8B | FP8 skipped (no native FP8 in current MUSA hardware); compressed-tensors W8A8/W8A16 candidate; AWQ / GPTQ pending validation |
| Suite D | Long context ~28K input, Llama-3.1-8B | Reduce `max_num_seqs` and `gpu_memory_utilization` |
| Suite E | Multi-chip scaling, Llama-3-8B | Validates MCCL tensor parallelism |
| Suite F | Consumer/edge, Qwen2.5-0.5B | Recommended starting point for S4000 single-card systems |

## Hardware compatibility

| GPU | BF16 | TP via MCCL | FP8 | Notes |
|-----|------|-------------|-----|-------|
| MTT S5000 | ✅ | ✅ | ❌ | Recommended public reference target (FA3 via MATE) |
| MTT S4000 | ✅ | ✅ | ❌ | Validated path with PyTorch SDPA-based FlashAttention |
| MTT S3000 | ⚠️ | ⚠️ | ❌ | May work via `--enforce-eager`; not the public reference |
| MTT S80 | ⚠️ | — | ❌ | Consumer card; treat as best-effort |

## Prerequisites

You must install the MUSA stack in this exact order — Python packages alone
are not sufficient:

**1. MUSA toolkit + driver**

Match the toolkit version to your card firmware. Reference:
<https://developer.mthreads.com/musa/>

**2. PyTorch with MUSA support (torch + torchada)**

The recommended path is the official Moore Threads container, which ships a
pre-built `torch==2.7.1` together with `torchada` and `pymtml`. See:

```bash
docker pull sh-harbor.mthreads.com/mcctest/musa-compile:rc4.3.3-torch2.7-20251120
```

**3. Runner dependencies**

Inside the MUSA container:

```bash
pip install -r runners/moorethreads_vllm_musa_57ff5443/requirements.txt
```

This installs `vllm-musa==0.1.1` which auto-pulls a validated vLLM core
(`0.10.1.1` by default). To use vLLM `0.13.0` instead (V1-only engine):

```bash
pip install vllm==0.13.0 --no-deps --upgrade
pip install 'depyf==0.20.0' 'llguidance>=1.3.0,<1.4.0' \
            'lm-format-enforcer==0.11.3' 'outlines_core==0.2.11' \
            'xgrammar==0.1.27' 'compressed-tensors==0.12.2'
```

## Required environment variables

```bash
# Device visibility (works like CUDA_VISIBLE_DEVICES)
export MUSA_VISIBLE_DEVICES=0,1,2,3

# Recommended for multi-process workers (TP > 1)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## Basic usage

```bash
# Verify the plugin is loaded before running anything else
python -c "from vllm_musa_platform import musa_platform_plugin; print('ok')"

# Suite F (single-card S4000 / S5000)
python run.py --runner moorethreads_vllm_musa_57ff5443 --suite suite_F

# Suite A (single-card datacenter benchmark)
python run.py --runner moorethreads_vllm_musa_57ff5443 --suite suite_A

# Multi-card tensor parallelism (e.g. 8 x S5000 on a single host)
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python run.py --runner moorethreads_vllm_musa_57ff5443 \
    --suite suite_B \
    --tensor-parallel-size 8

# Local model cache
python run.py --runner moorethreads_vllm_musa_57ff5443 \
    --suite suite_A \
    --model-path /data/models/Meta-Llama-3-8B-Instruct
```

## Runner config

Copy the example config and adjust for your hardware:

```bash
cp configs/runner_configs/runner_moorethreads_vllm_musa_57ff5443.yaml.example \
   configs/runner_configs/runner_moorethreads_vllm_musa_57ff5443.yaml
```

Key settings:

| Field | Default | Notes |
|-------|---------|-------|
| `tensor_parallel_size` | 1 | Number of MUSA GPUs for tensor parallelism |
| `enforce_eager` | false | Disable CUDA-graph / compilation; useful for pre-S4000 cards or while debugging |
| `max_num_seqs` | 256 | Max concurrent sequences; reduce on lower-memory cards |
| `gpu_memory_utilization` | 0.85 | Fraction of HBM reserved for KV cache; reduce if OOM |

## Triton / kernel compilation errors

If you encounter errors during Triton graph capture on first request,
disable graph capture with `--enforce-eager`:

```bash
python run.py --runner moorethreads_vllm_musa_57ff5443 \
    --suite suite_F --enforce-eager
```

Or set persistently in the runner config YAML:

```yaml
enforce_eager: true
```

## HBM OOM errors

Reduce `gpu_memory_utilization` and/or `max_num_seqs`, either globally or
per-suite (Suite D is the most memory-hungry due to long-context inputs):

```yaml
gpu_memory_utilization: 0.80
max_num_seqs: 128

suites:
  suite_D:
    max_num_seqs: 32
    gpu_memory_utilization: 0.78
```

## Known gaps (pre-smoke-test)

The following items are placeholders and **must be re-validated** on real
S4000 / S5000 hardware:

- **Memory peak**: relies on `torch.cuda.max_memory_allocated()` which
  torchada aliases to MUSA. If this returns 0 or `None`, fall back to
  `pymtml.mtmlDeviceGetMemoryInfo()`.
- **MCCL teardown**: assumes the same `cleanup_dist_env_and_memory` entry
  point as upstream vLLM. If MCCL leaves a hanging process group, the
  fallback path explicitly destroys the torch.distributed group.
- **Quantization**: `SUPPORTED_QUANTIZATION_BACKENDS` currently lists only
  `compressed-tensors`. AWQ / GPTQ-Marlin / FP8 are intentionally excluded
  until kernel coverage on MUSA is confirmed.
- **Precision detection**: `_get_chip_count()` prefers `pymtml` over
  `torch.cuda.device_count()`. On hosts where pymtml is missing this may
  miscount; in that case the torch fallback should still work because
  torchada provides `torch.cuda.device_count()`.

## Requirements

See `requirements.txt` for the pinned plugin / extras list. The heavy
dependencies (torch + torchada + MUSA toolkit) must come from the Moore
Threads container; do not install them from PyPI.

Minimum environment:
- Moore Threads MTT S4000 or newer (S3000 / S80 best-effort)
- MUSA toolkit + driver matching card firmware
- torch 2.7.1 (Moore Threads MUSA build) + torchada ≥ 0.1.9
- Python 3.10+
- vllm-musa 0.1.1 (vLLM core 0.10.1.1 or 0.13.0)
