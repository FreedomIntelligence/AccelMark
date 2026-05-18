# nvidia_vllm020_0f6c56e4 — NVIDIA vLLM Runner (0.20.x)

AccelMark reference runner for NVIDIA GPUs running **vLLM 0.20.x**.

Supersedes [`nvidia_vllm_47f5d58e`](../nvidia_vllm_47f5d58e/) (vLLM 0.7.3). Use the predecessor for CUDA 11.8 / legacy stacks; use this runner for Ampere+ datacenter GPUs with CUDA 12.8 or 13.0.

## Supported suites

| Suite | Description | Notes |
|-------|-------------|-------|
| Suite A | Single-chip, Llama-3-8B | Speculative and burst extra scenarios |
| Suite B | Multi-chip, Llama-3-70B | Requires 4× A100/H100 or equivalent |
| Suite C | Quantization, Llama-3.1-8B | **Requires `enforce_eager: true` in runner config** — see below |
| Suite D | Long context ~28K input | `max_model_len` 30,208 |
| Suite E | Multi-chip scaling, Llama-3-8B | NVLink recommended |
| Suite F | Consumer/edge, Qwen2.5-0.5B | Pre-Ampere: use predecessor + `--enforce-eager` |
| Suite G | MoE multi-chip, Mixtral-8x7B | ≥2× A100-80GB |

## What changed vs nvidia_vllm_47f5d58e

| Area | 0.7.3 (predecessor) | 0.20.x (this runner) |
|---|---|---|
| Default CUDA | 12.1 | **13.0** (12.8 via `PYTORCH_INDEX`) |
| PyTorch | 2.5.1 | **2.11** (pulled by vLLM) |
| Python | 3.10+ | **3.10–3.12** |
| Transformers | v4.57 | vLLM-pinned (see `result.json` version string) |
| FlashAttention | FA2 | FA4 (MLA prefill default on supported models) |
| Quantization | fp8, compressed-tensors, gptq_marlin | + **turboquant** |
| Model runner | V1 | V2 |

Release notes: [v0.20.0](https://github.com/vllm-project/vllm/releases/tag/v0.20.0) · [v0.20.1](https://github.com/vllm-project/vllm/releases/tag/v0.20.1).

## Installation

### Prerequisites

- NVIDIA GPU, compute capability ≥ 7.0 (Volta+; Ampere+ recommended)
- **CUDA 13.0** driver/runtime (default for this stack), or **CUDA 12.8** via PyTorch index below
- **Python 3.10, 3.11, or 3.12** (not 3.13+ until vLLM supports it)
- A clean virtualenv/conda env if upgrading from `vllm==0.7.3` (mixed installs break imports)

### Recommended: `install.sh`

From the AccelMark repo root:

```bash
# Create and activate a fresh env (example)
conda create -n accel python=3.12 -y
conda activate accel

# Default install (CUDA 13.0 wheels from vLLM)
bash runners/nvidia_vllm020_0f6c56e4/install.sh
```

CUDA **12.8** hosts must point pip at the cu128 PyTorch index:

```bash
PYTORCH_INDEX=https://download.pytorch.org/whl/cu128 \
  bash runners/nvidia_vllm020_0f6c56e4/install.sh
```

`install.sh` reads versions from `requirements.txt` and installs in three stages (pip cannot resolve `vllm` and `mistral-common[image]` in one pass). **Do not** run `pip install -r requirements.txt` directly.

### Verify

```bash
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

### Manual install (equivalent to `install.sh`)

```bash
pip install mistral-common==1.11.2
pip install vllm==0.20.1    # add --extra-index-url if using PYTORCH_INDEX above
pip install "numpy>=1.26.0,<2.0" jsonschema psutil tqdm nvidia-ml-py PyYAML
```

### Submitter profile and local models

```bash
cp configs/submitter.yaml.example configs/submitter.yaml   # set submitted_by
cp configs/models_local.yaml.example configs/models_local.yaml   # optional local paths
```

## Usage

```bash
python run.py --runner nvidia_vllm020_0f6c56e4 --suite suite_A
python run.py --runner nvidia_vllm020_0f6c56e4 --suite suite_B --tensor-parallel-size 4
python run.py --runner nvidia_vllm020_0f6c56e4 --suite suite_C
```

Or invoke the runner directly:

```bash
python runners/nvidia_vllm020_0f6c56e4/runner.py --suite suite_F --scenario offline
```

## Runner config

```bash
cp configs/runner_configs/runner_nvidia_vllm020_0f6c56e4.yaml.example \
   configs/runner_configs/runner_nvidia_vllm020_0f6c56e4.yaml
```

Merge priority: CLI flags > suite-specific section > global defaults.

### Suite C — quantization (`enforce_eager` required)

vLLM 0.20 enables CUDA graphs by default. With `compressed-tensors` checkpoints (FP8, W8A8, W8A16), graphs can produce **repetitive garbage output**: offline throughput looks normal but MMLU accuracy drops to ~0.

The example config sets this only for Suite C so other suites keep CUDA graphs:

```yaml
suites:
  suite_C:
    enforce_eager: true
```

CLI override: `--enforce-eager`. Without it, Suite C accuracy results are invalid even if throughput is high.

### Optional `engine_kwargs` (0.20)

```yaml
engine_kwargs:
  attention_backend: FLASH_ATTN_4
  # compilation_config:
  #   cudagraph_mode: full_and_piecewise
  # kv_cache_dtype: turboquant   # experimental; suite C
```

See [vLLM EngineArgs](https://docs.vllm.ai/en/latest/api/vllm/engine/arg_utils.html).

## Troubleshooting

### Large-memory GPUs (H20, A100 80GB) — SIGFPE / silent crash

Symptom: subprocess exits with `SIGFPE (return code -8)` after model load or on first batch.

```bash
pip install --upgrade nvidia-cublas-cu13
```

On CUDA 12.8 stacks use `nvidia-cublas-cu12` instead. Details: [predecessor README](../nvidia_vllm_47f5d58e/README.md#large-memory-gpus-h20-a100-80-gb-etc).

### Pre-Ampere (V100, T4, RTX 20xx)

This runner targets Ampere+ with CUDA 12.8/13.0. For Volta/Turing, use [`nvidia_vllm_47f5d58e`](../nvidia_vllm_47f5d58e/) with `--enforce-eager` (BF16→FP16 fallback, no CUDA graphs). See the predecessor README for Suite F / Suite A on V100.

### Suite C accuracy ~0 but offline OK

Enable `enforce_eager` for `suite_C` in the runner config (see above) and re-run the accuracy scenario.

## Hardware matrix

Full GPU compatibility table: [`nvidia_vllm_47f5d58e/README.md`](../nvidia_vllm_47f5d58e/README.md#hardware-compatibility).

## Files

| File | Purpose |
|------|---------|
| `runner.py` | Runner implementation |
| `meta.json` | Runner metadata and suite support |
| `requirements.txt` | Pinned dependency list (source of truth) |
| `install.sh` | Staged pip install |
