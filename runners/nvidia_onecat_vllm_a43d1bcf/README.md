# nvidia_onecat_vllm_a43d1bcf — 1Cat-vLLM Runner (Tesla V100 / SM70)

AccelMark runner for **Tesla V100 / V100S only**, using
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) — the community vLLM fork
that re-enables modern AWQ 4-bit serving and FlashAttention on Volta GPUs
(SM70).

> **Hardware scope:** This runner is intentionally narrow. On Ampere
> (A100/A800/A10/L4/4090/etc.) or newer, use the upstream
> `nvidia_vllm_*` runner — 1Cat-vLLM's kernels are tuned for SM70 and
> provide no benefit on later architectures.

> **Status:** Committed without an end-to-end validation run yet. The runner
> code is a thin specialisation of the upstream NVIDIA vLLM runner (only
> capability flags + attention-backend default differ), so existing test
> coverage of the parent runner applies. Plan to add a reference
> `Tesla V100-SXM2-32GBx4 suite_B` result once a target box is available.

## Why 1Cat-vLLM exists

| Pain on stock vLLM + V100 | 1Cat-vLLM's fix |
|---|---|
| AWQ kernels require SM75+ | Integrated lmdeploy TurboMind WMMA kernels for SM70 |
| FlashAttention 2/3 require Ampere+ | Custom `FLASH_ATTN_V100` Volta backend |
| Qwen3.5 / Qwen3.6 dense + MoE not loadable | Model configs and runtime fixes shipped in fork |
| Long-context paged-prefill stability | SM70-specific MLA/GDN runtime fixes |
| FP8 KV cache | `fp8_e5m2` (experimental) on V100 FA path |

For full release notes see
<https://github.com/1CatAI/1Cat-vLLM> RELEASE_NOTES_1.0.0.md.

## Defaults this runner injects

| Knob | Default | Where set | Why |
|---|---|---|---|
| `attention_backend` | `FLASH_ATTN_V100` | `load_model()` if not already specified | 1Cat-vLLM's recommended V100 path |
| `SUPPORTED_PRECISIONS` | `["fp16", "fp32"]` | class attribute | V100 has no BF16 |
| `SUPPORTED_QUANTIZATION_BACKENDS` | `["awq"]` | class attribute | 1Cat's headline kernel; other formats not validated on this stack |
| `max_num_seqs` | `1` | runner config default | 1Cat 1.0.0 public default — 256K context on V100 |
| `gpu_memory_utilization` | `0.88` | runner config default | 1Cat 1.0.0 public default |

To opt into the MTP + prefix-cache profile (Qwen3.6-27B-AWQ), bump
`max_num_seqs` to `4` and pass `speculative_config` via the runner config
`engine_kwargs` — see the example config file.

## Supported suites

| Suite | Recommendation |
|-------|---------------|
| Suite A — Llama-3-8B 1× | Runs, but vanilla `nvidia_vllm_47f5d58e --enforce-eager` already covers this. Use 1Cat only if you want the FA-V100 attention path. |
| Suite B — Llama-3-70B multi-chip | **Primary target.** Recommended `--tensor-parallel-size 4`. |
| Suite C — Quantization | Restricted to AWQ — this is where 1Cat shines. |
| Suite D — Long context (~28K) | **Primary target.** `FLASH_ATTN_V100` is the only V100-friendly long-context path. |
| Suite E — Scaling | Same considerations as Suite B; useful for measuring how 1Cat's MCCL-equivalent scales. |
| Suite F — Qwen2.5-0.5B edge | Not interesting on V100 — the model fits trivially; use upstream runner. |
| Suite G — MoE | Sweet spot — `Qwen3.6-35B-A3B-AWQ`, `Qwen3.5-122B-A10B-AWQ` are exactly the validated MoE models in 1Cat 1.0.0. |

## Environment setup

1Cat-vLLM 1.0.0 ships **prebuilt wheels only** (no PyPI `vllm`). Install the
wheels **before** `requirements.txt` — the extras file intentionally omits
`torch` / `vllm` so it does not fight the cu128 index used by the wheels.

### Validated stack (1Cat-vLLM 1.0.0)

| Component | Version |
|-----------|---------|
| OS | Ubuntu **24.04** (glibc ≥ 2.38) |
| Python | **3.12** (`cp312` wheels only) |
| CUDA | **12.8** toolkit + matching driver (570.x recommended) |
| PyTorch | **2.9.1+cu128** (pulled in by the wheels) |
| GPU | Tesla V100 / V100S (SM70) |

Upstream reference: [1Cat-vLLM releases](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.0.0)
and [installation guide](https://github.com/1CatAI/1Cat-vLLM#quick-start).

### Ubuntu 22.04 and other older hosts

The release wheels are linked against **glibc 2.38**. On Ubuntu 22.04 (glibc
2.35), `pip install` may succeed but `import vllm` fails with
`GLIBC_2.38 not found`. Options:

- Run on **Ubuntu 24.04** (bare metal or VM), or
- Use a **glibc ≥ 2.38 container** on the host (see the [1Cat-vLLM Docker
  notes](https://github.com/1CatAI/1Cat-vLLM#docker-deployment) — build/run
  on a machine where the Docker daemon is available; nested dev containers
  without `docker.sock` bind-mount usually cannot host Docker), or
- **Build from source** on your host glibc (see 1Cat-vLLM “Source build”).

### Install steps

From the AccelMark repo root, in a fresh **Python 3.12** environment:

```bash
# 1. CUDA 12.8 toolkit + driver
#    https://developer.nvidia.com/cuda-12-8-0-download-archive

conda create -y -n onecat-vllm python=3.12
conda activate onecat-vllm
python -m pip install --upgrade pip setuptools wheel

# 2. 1Cat-vLLM wheels (install BOTH together — do not use PyPI vllm)
python -m pip install --prefer-binary --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/flash_attn_v100-1.0.0-cp312-cp312-linux_x86_64.whl" \
    "https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/vllm-1.0.0-cp312-cp312-linux_x86_64.whl"

# 3. AccelMark runner extras only
python -m pip install -r runners/nvidia_onecat_vllm_a43d1bcf/requirements.txt
```

Do **not** install `vllm` from PyPI afterward — it will replace the fork.
Run benchmarks from a directory **outside** a cloned 1Cat-vLLM source tree so
Python does not import the local `vllm/` package instead of the wheel.

## Smoke test the install

```bash
python - <<'PY'
import torch, vllm
print("torch:", torch.__version__, "  vllm:", vllm.__version__)
try:
    import flash_attn_v100_cuda  # SM70 FA kernels
    print("flash_attn_v100: ok")
except Exception as e:
    print("flash_attn_v100: MISSING ->", e)
PY
```

`flash_attn_v100` MUST be importable — if it isn't, you accidentally
installed plain vLLM from PyPI; reinstall from the 1Cat release wheels above.

## Basic usage

```bash
# Suite D (long-context) on 4 x V100 32 GB
python run.py --runner nvidia_onecat_vllm_a43d1bcf \
    --suite suite_D \
    --tensor-parallel-size 4

# Suite C with AWQ (Qwen3.5-27B-AWQ as the validation model)
python run.py --runner nvidia_onecat_vllm_a43d1bcf \
    --suite suite_C \
    --tensor-parallel-size 4 \
    --model-path /data/models/Qwen3.5-27B-AWQ

# Override attention backend (rare — for benchmarking vs Triton fallback)
python run.py --runner nvidia_onecat_vllm_a43d1bcf \
    --suite suite_B \
    --tensor-parallel-size 4 \
    # Then set attention_backend in your runner config engine_kwargs.
```

## Runner config

Copy the example:

```bash
cp configs/runner_configs/runner_nvidia_onecat_vllm_a43d1bcf.yaml.example \
   configs/runner_configs/runner_nvidia_onecat_vllm_a43d1bcf.yaml
```

Key defaults differ from the upstream NVIDIA runner:

| Field | 1Cat default | Upstream default | Notes |
|-------|--------------|------------------|-------|
| `max_num_seqs` | 1 | 512 | 256K context demands very tight KV cache budget |
| `gpu_memory_utilization` | 0.88 | 0.90 | Matches 1Cat 1.0.0 public reference |
| `engine_kwargs.attention_backend` | `FLASH_ATTN_V100` (auto) | — | Auto-injected unless overridden |

## Known gaps (pre-smoke-test)

- The Volta CUDA-graph capture path needs validation under
  `--scenario sustained`. If startup hangs on the first request, set
  `enforce_eager: true` in your runner config.
- The accuracy gate uses the suite's stock prompts — on AWQ checkpoints
  the gate threshold may be too tight; the suite spec already allows
  per-format thresholds (Suite C) so this is mostly relevant on Suite A/D.
- MTP / speculative profiles are documented in 1Cat 1.0.0 but not
  exercised here yet; flat speculative keys in `_precision_engine_kwargs`
  are still forwarded as `speculative_config` by `benchmark_runner.py`,
  the same as for the upstream runner.

## Requirements

See `requirements.txt`. The heavy dependencies (`torch`, `flash_attn_v100`,
`vllm` fork) MUST come from the 1Cat-vLLM release wheels — do not install
upstream `vllm` from PyPI.
