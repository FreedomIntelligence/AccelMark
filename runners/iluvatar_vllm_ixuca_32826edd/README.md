# iluvatar_vllm_ixuca_32826edd — Iluvatar CoreX Runner (vllm-ixuca)

AccelMark runner for Iluvatar CoreX GPUs via the **IXUCA** software stack
(vllm-ixuca).

## Supported suites

| Suite | Description | Notes |
|-------|-------------|-------|
| Suite A | Single-chip, Llama-3-8B | Pending — not yet smoke-tested |
| Suite B | Multi-chip, Llama-3-70B | ixCCL tensor parallelism |
| Suite C | Quantization, Llama-3.1-8B | FP8 skipped; GPTQ/INT8; W8A8/W8A16 via compressed-tensors |
| Suite D | Long context ~28K input | Reduce `max_num_seqs` / `gpu_memory_utilization` in runner config |
| Suite E | Multi-chip scaling, Llama-3-8B | ixCCL tensor parallelism |
| Suite F | Edge, Qwen2.5-0.5B | Pending |
| Suite G | MoE multi-chip, Mixtral-8x7B | Pending |
| Suite H | Mistral-7B SWA | Pending |

## Hardware compatibility

| GPU | BF16 | FP16 | Multi-chip TP | FP8 | Notes |
|-----|------|------|---------------|-----|-------|
| CoreX 智铠 MR-V100 | ✅ | ✅ | ✅ (ixCCL) | ❌ | torch_ixuca required |

FP8 is **not** supported on 智铠100. Marlin kernels are CUDA-only; use `gptq`
(IXUCA kernel), not `gptq_marlin`. IXUCA presents the accelerator as
CUDA-compatible, so no explicit `device=` flag is required.

## Prerequisites

Install in this order — **do not** `pip install torch` or `vllm` from PyPI on
a bare Linux host:

1. **IXUCA SDK + torch_ixuca** — obtained from Iluvatar (proprietary; no public PyPI).
2. **vllm-ixuca** — the vLLM adaptation for IXUCA.
3. **Runner dependencies**:

   ```bash
   pip install -r runners/iluvatar_vllm_ixuca_32826edd/requirements.txt
   ```

## Smoke test

```bash
python runners/iluvatar_vllm_ixuca_32826edd/test_smoke.py
python runners/iluvatar_vllm_ixuca_32826edd/test_smoke.py /path/to/model
```

## Usage

```bash
python run.py --runner iluvatar_vllm_ixuca_32826edd --suite suite_A --precision BF16

# Multi-chip tensor parallelism (ixCCL)
python run.py --runner iluvatar_vllm_ixuca_32826edd \
  --suite suite_B --tensor-parallel-size 8
```

Optional runner config (copy and edit):

```bash
cp configs/runner_configs/runner_iluvatar_vllm_ixuca_32826edd.yaml.example \
   configs/runner_configs/runner_iluvatar_vllm_ixuca_32826edd.yaml
```

| Field | Default | Notes |
|-------|---------|-------|
| `tensor_parallel_size` | 1 | ixCCL tensor parallelism |
| `enforce_eager` | false | Only if graph capture errors |
| `max_num_seqs` | 512 | Lower on small HBM |
| `gpu_memory_utilization` | 0.90 | Lower if OOM |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `torch.cuda` missing / no device | Ensure `torch_ixuca` is installed and loadable |
| OOM | Lower `gpu_memory_utilization` / `max_num_seqs` |
| Graph capture errors | `--enforce-eager` or `enforce_eager: true` in runner YAML |
| FP8 errors | FP8 unsupported — use BF16/FP16 |

## Requirements

See `requirements.txt` for AccelMark extras. The IXUCA SDK, torch_ixuca, and
vllm-ixuca are installed per Iluvatar's distribution (not from this file).
