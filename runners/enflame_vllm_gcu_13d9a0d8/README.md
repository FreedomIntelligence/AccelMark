# enflame_vllm_gcu_13d9a0d8 — Enflame GCU Runner (vllm-gcu)

AccelMark runner for Enflame GCU accelerators using
[vllm-gcu](https://github.com/EnflameTechnology/vllm-gcu).

## Supported suites

| Suite | Description | Notes |
|-------|-------------|-------|
| Suite A | Single-chip, Llama-3-8B | Pending — not yet smoke-tested |
| Suite B | Multi-chip, Llama-3-70B | GCU interconnect TP/PP |
| Suite C | Quantization, Llama-3.1-8B | FP8 skipped; GPTQ/AWQ; W8A8/W8A16 via compressed-tensors |
| Suite D | Long context ~28K input | Reduce `max_num_seqs` / `gpu_memory_utilization` in runner config |
| Suite E | Multi-chip scaling, Llama-3-8B | GCU interconnect |
| Suite F | Edge, Qwen2.5-0.5B | Pending |
| Suite G | MoE multi-chip, Mixtral-8x7B | Pending |
| Suite H | Mistral-7B SWA | Pending |

## Hardware compatibility

| Accelerator | BF16 | FP16 | Multi-chip TP | FP8 | Notes |
|-------------|------|------|---------------|-----|-------|
| Enflame 云燧 S60 GCU | ✅ | ✅ | ✅ (GCU interconnect) | ❌ | TopsRider i3x 3.6+ |

FP8 is **not** supported on S60 GCU. Marlin kernels are CUDA-only; use `gptq`
(GCU kernel), not `gptq_marlin`. vllm-gcu **requires** `device="gcu"` — the
runner sets this automatically.

## Prerequisites

Install in this order — **do not** `pip install torch` or `vllm` from PyPI on
a bare Linux host:

1. **TopsRider runtime + torch_gcu** — per
   [vllm-gcu](https://github.com/EnflameTechnology/vllm-gcu) (proprietary).
2. **vllm-gcu** — build from source (`setup.py bdist_wheel`) or use the
   provided Docker image.
3. **Runner dependencies**:

   ```bash
   pip install -r runners/enflame_vllm_gcu_13d9a0d8/requirements.txt
   ```

## Smoke test

```bash
python runners/enflame_vllm_gcu_13d9a0d8/test_smoke.py
python runners/enflame_vllm_gcu_13d9a0d8/test_smoke.py /path/to/model
```

## Usage

```bash
python run.py --runner enflame_vllm_gcu_13d9a0d8 --suite suite_A --precision BF16

# Multi-chip tensor parallelism
python run.py --runner enflame_vllm_gcu_13d9a0d8 \
  --suite suite_B --tensor-parallel-size 4
```

Optional runner config (copy and edit):

```bash
cp configs/runner_configs/runner_enflame_vllm_gcu_13d9a0d8.yaml.example \
   configs/runner_configs/runner_enflame_vllm_gcu_13d9a0d8.yaml
```

| Field | Default | Notes |
|-------|---------|-------|
| `tensor_parallel_size` | 1 | GCU interconnect tensor parallelism |
| `enforce_eager` | false | Only if graph capture errors |
| `max_num_seqs` | 512 | Lower on small HBM |
| `gpu_memory_utilization` | 0.90 | Lower if OOM |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Expected gcu device, got cuda:0` | Use this runner (`device="gcu"`) |
| OOM | Lower `gpu_memory_utilization` / `max_num_seqs` |
| Graph capture errors | `--enforce-eager` or `enforce_eager: true` in runner YAML |
| FP8 errors | FP8 unsupported — use BF16/FP16 |

## Requirements

See `requirements.txt` for AccelMark extras. vllm-gcu, torch_gcu, and the
TopsRider driver are installed per the upstream vllm-gcu guide (not from this file).
