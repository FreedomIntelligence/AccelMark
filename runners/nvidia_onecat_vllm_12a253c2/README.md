# nvidia_onecat_vllm_12a253c2 — 1Cat-vLLM Runner (Tesla V100 / SM70)

AccelMark runner for **Tesla V100 / V100S only**, using
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (community vLLM fork for Volta).

> **Hardware:** Use this runner only on V100 / V100S (SM70). On Ampere or newer,
> use upstream `nvidia_vllm_*`.

> **Third-party software:** 1Cat-vLLM is maintained by [1CatAI](https://github.com/1CatAI/1Cat-vLLM)
> under its own license. AccelMark ships only the thin `runner.py` wrapper; install
> 1Cat-vLLM separately as described below.

## Why 1Cat-vLLM

| Limitation on stock vLLM + V100 | 1Cat-vLLM |
|--------------------------------|-----------|
| AWQ kernels need SM75+ | SM70 AWQ via lmdeploy TurboMind |
| FlashAttention 2/3 need Ampere+ | `FLASH_ATTN_V100` backend |
| Qwen3.5 / Qwen3.6 on V100 | Fork model/runtime fixes |
| Long-context on Volta | SM70 paged-attention path |

Release notes: [1Cat-vLLM v1.0.0](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.0.0).

## Runner defaults (code)

| Setting | Default |
|---------|---------|
| `attention_backend` | `FLASH_ATTN_V100` (auto unless overridden) |
| `SUPPORTED_PRECISIONS` | `fp16`, `fp32` (no BF16 on V100) |
| `SUPPORTED_QUANTIZATION_BACKENDS` | `awq` only |
| `max_num_seqs` | `512` global default (same as upstream vLLM); use `1` for suite D / long-context |
| `gpu_memory_utilization` | `0.90` |

## Supported suites

| Suite | Notes |
|-------|-------|
| A | Runs on 1× V100; upstream `nvidia_vllm_*` + `--enforce-eager` is often enough |
| B | **Primary** — use `--tensor-parallel-size 4` on 4× V100 32GB |
| C | **Primary** — AWQ |
| D | **Primary** — long context + `FLASH_ATTN_V100` |
| E | Multi-chip scaling (same TP guidance as B) |
| F | Not recommended (edge model; use upstream runner) |
| G | **Primary** — MoE + AWQ (Qwen3.5/3.6 class models) |

---

## Environment setup

### Reference stack (1Cat-vLLM 1.0.0)

| Component | Version |
|-----------|---------|
| GPU | Tesla V100 / V100S (SM70) |
| Python | **3.12** (`cp312` wheels only) |
| CUDA toolkit | **12.8** |
| Driver | 570.x recommended (CUDA 12.8) |
| PyTorch | **2.9.1+cu128** (from 1Cat wheels or build env) |

### Path A — Prebuilt wheels (Ubuntu 24.04+, glibc ≥ 2.38)

Official wheels require **glibc 2.38+** (e.g. Ubuntu 24.04). On Ubuntu 22.04,
`pip install` may succeed but `import vllm` fails with `GLIBC_2.38 not found`
— use Path B instead.

```bash
conda create -y -n onecat-vllm python=3.12
conda activate onecat-vllm
python -m pip install --upgrade pip setuptools wheel

# Install BOTH wheels together — never `pip install vllm` from PyPI
python -m pip install --prefer-binary --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/flash_attn_v100-1.0.0-cp312-cp312-linux_x86_64.whl" \
    "https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/vllm-1.0.0-cp312-cp312-linux_x86_64.whl"

cd /path/to/AccelMark
pip install -r runners/nvidia_onecat_vllm_12a253c2/requirements.txt
```

### Path B — Build from source (Ubuntu 22.04 / glibc 2.35)

Build on the **host glibc** so binaries link against 2.35. Typical AutoDL /
Ubuntu 22.04 V100 boxes use this path.

**Prerequisites:** CUDA 12.8 toolkit (`nvcc` on PATH), conda Python 3.12, ~20GB
free disk for build tree + wheels.

```bash
conda create -y -n onecat-vllm python=3.12
conda activate onecat-vllm
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=6
export PIP_CACHE_DIR=/path/to/fast/disk/pip-cache   # optional

git clone --depth 1 --branch v1.0.0 https://github.com/1CatAI/1Cat-vLLM.git
cd 1Cat-vLLM
pip install -r requirements/build.txt -r requirements/cuda.txt -r requirements/common.txt
pip install cmake build ninja

DIST=/path/to/dist-cu128-sm70-v1.0.0
mkdir -p "$DIST"

# 1) flash_attn_v100 wheel
pushd flash-attention-v100
python -m build --wheel --no-isolation --outdir "$DIST"
popd

# 2) vllm wheel (30–90 min on V100 host)
export VLLM_TARGET_DEVICE=cuda
python -m build --wheel --no-isolation --outdir "$DIST"

# 3) Install — run from /tmp so Python does not import the source tree
pip install "$DIST"/flash_attn_v100-*.whl
cd /tmp && pip install --no-deps --force-reinstall "$DIST"/vllm-*.whl

cd /path/to/AccelMark
pip install -r runners/nvidia_onecat_vllm_12a253c2/requirements.txt
```

Do **not** run AccelMark from inside the cloned `1Cat-vLLM/` directory; Python
may import the local `vllm/` package instead of the installed wheel.

### Smoke test

Run from `/tmp` or the AccelMark repo root (not inside `1Cat-vLLM/`):

```bash
python - <<'PY'
import torch, vllm
print("torch:", torch.__version__, "vllm:", vllm.__version__)
import flash_attn_v100_cuda
print("flash_attn_v100: ok")
from vllm import LLM
print("LLM import: ok")
PY
```

---

## AccelMark runner config (required on V100)

Copy and edit:

```bash
cp configs/runner_configs/runner_nvidia_onecat_vllm_12a253c2.yaml.example \
   configs/runner_configs/runner_nvidia_onecat_vllm_12a253c2.yaml
```

**Single V100 32GB** — recommended `engine_kwargs` (avoids SM70
`Shared memory exceeds 96KB` in `prefill_paged_fwd`):

```yaml
tensor_parallel_size: 1
max_num_seqs: 512
gpu_memory_utilization: 0.90
engine_kwargs:
  enable_prefix_caching: false
  enable_chunked_prefill: false
  kv_cache_auto_trim_ratio: 0.0

suites:
  suite_D:
    max_num_seqs: 1
    gpu_memory_utilization: 0.85
```

If it still crashes, export before `python run.py`:

```bash
export VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1
```

That forces the slower paged-KV gather fallback instead of `prefill_paged_fwd`.

**4× V100 32GB** — set `tensor_parallel_size: 4`; keep the same `engine_kwargs`
unless you are deliberately testing 1Cat's MTP / prefix-cache profile (see
example file comments).

Other tuning:

| Symptom | Try |
|---------|-----|
| `Shared memory exceeds 96KB` | `enable_chunked_prefill: false` + `enable_prefix_caching: false` (above); then `export VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1` |
| First request hangs (CUDA graph) | `enforce_eager: true` or `--enforce-eager` |
| OOM at engine init | Lower `gpu_memory_utilization` (e.g. `0.85`) |
| `GLIBC_2.38 not found` | Path B source build, or Ubuntu 24.04+ |

---

## Basic usage

```bash
cp configs/submitter.yaml.example configs/submitter.yaml   # once
cp configs/models_local.yaml.example configs/models_local.yaml   # map local model paths

export PYTHONPATH=/path/to/AccelMark   # if pip install -e . is unavailable

# Suite A smoke (1× V100)
python run.py --runner nvidia_onecat_vllm_12a253c2 \
    --suite suite_A --scenario accuracy --tensor-parallel-size 1

# Suite B (4× V100)
python run.py --runner nvidia_onecat_vllm_12a253c2 \
    --suite suite_B --tensor-parallel-size 4
```

---

## Known limitations

- Prefix caching and **chunked prefill** (even with prefix caching off) can hit the
  `prefill_paged_fwd` kernel (>96KB shared memory on SM70). Disable both in config;
  use `VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1` if needed (see above).
- `max_num_seqs: 1` limits batch throughput vs upstream vLLM defaults — intentional
  for 1Cat's long-context V100 profile.
- Suite F is marked unsupported in `meta.json` (use upstream runner on V100 if needed).
- End-to-end validation on 4× V100 reference hardware is still community-pending in
  `meta.json`; single-GPU smoke (Suite A accuracy) has been exercised on V100 32GB.

## Requirements

See `requirements.txt`. Install `torch`, `flash_attn_v100`, and the `vllm` fork
from 1Cat-vLLM **before** the AccelMark extras file. Do not install upstream
`vllm` from PyPI after the fork.
