# hygon_vllm_dcu_94fd7f1e — Hygon DCU Runner (vLLM-DCU / ROCm)

AccelMark runner for Hygon DCU accelerators via the **ROCm-compatible DTK**
stack (vLLM-DCU).

## Supported suites

| Suite | Description | Notes |
|-------|-------------|-------|
| Suite A | Single-chip, Llama-3-8B | Pending — not yet smoke-tested |
| Suite B | Multi-chip, Llama-3-70B | RCCL tensor parallelism |
| Suite C | Quantization, Llama-3.1-8B | FP8 skipped; GPTQ/INT8; W8A8/W8A16 via compressed-tensors |
| Suite D | Long context ~28K input | Reduce `max_num_seqs` / `gpu_memory_utilization` in runner config |
| Suite E | Multi-chip scaling, Llama-3-8B | RCCL tensor parallelism |
| Suite F | Edge, Qwen2.5-0.5B | Pending |
| Suite G | MoE multi-chip, Mixtral-8x7B | Pending |
| Suite H | Mistral-7B SWA | Pending |

## Hardware compatibility

| Accelerator | BF16 | FP16 | FP32 | Multi-chip TP | FP8 | Notes |
|-------------|------|------|------|---------------|-----|-------|
| Hygon DCU Z100 / K100 (深算系列) | ✅ | ✅ | ✅ | ✅ (RCCL) | ❌ | Hygon DTK (ROCm-compatible) |

FP8 is **not** supported on current DCU hardware. Marlin kernels are CUDA-only;
use `gptq` (DCU kernel), not `gptq_marlin`. DCU is enumerated as a ROCm/HIP
device, so no explicit `device=` flag is required.

## Prerequisites

Install in this order — **do not** `pip install torch` or `vllm` from PyPI on
a bare Linux host:

1. **Hygon DTK (ROCm-compatible) + torch-rocm** — obtained from Hygon/Sugon
   (proprietary; no public PyPI).
2. **vLLM-DCU** — vLLM built against the DCU ROCm backend.
3. **Runner dependencies**:

   ```bash
   pip install -r runners/hygon_vllm_dcu_94fd7f1e/requirements.txt
   ```

## Smoke test

```bash
python runners/hygon_vllm_dcu_94fd7f1e/test_smoke.py
python runners/hygon_vllm_dcu_94fd7f1e/test_smoke.py /path/to/model
```

## Usage

```bash
python run.py --runner hygon_vllm_dcu_94fd7f1e --suite suite_A --precision BF16

# Multi-chip tensor parallelism (RCCL)
python run.py --runner hygon_vllm_dcu_94fd7f1e \
  --suite suite_B --tensor-parallel-size 8
```

Optional runner config (copy and edit):

```bash
cp configs/runner_configs/runner_hygon_vllm_dcu_94fd7f1e.yaml.example \
   configs/runner_configs/runner_hygon_vllm_dcu_94fd7f1e.yaml
```

| Field | Default | Notes |
|-------|---------|-------|
| `tensor_parallel_size` | 1 | RCCL tensor parallelism |
| `enforce_eager` | false | Only if graph capture errors |
| `max_num_seqs` | 512 | Lower on small HBM |
| `gpu_memory_utilization` | 0.90 | Lower if OOM |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `hipErrorNoDevice` / no DCU | Ensure Hygon DTK + ROCm runtime are sourced |
| OOM | Lower `gpu_memory_utilization` / `max_num_seqs` |
| Graph capture errors | `--enforce-eager` or `enforce_eager: true` in runner YAML |
| FP8 errors | FP8 unsupported — use BF16/FP16/FP32 |

## Requirements

See `requirements.txt` for AccelMark extras. Hygon DTK, torch-rocm, and
vLLM-DCU are installed per Hygon/Sugon's distribution (not from this file).
