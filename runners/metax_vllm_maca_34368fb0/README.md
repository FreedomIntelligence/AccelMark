# metax_vllm_maca_34368fb0 — MetaX MXMACA Runner (vllm-metax)

AccelMark runner for MetaX MXMACA GPUs using **vllm-metax** (MacaRT-vLLM).

## Supported suites

| Suite | Description | Notes |
|-------|-------------|-------|
| Suite A | Single-chip, Llama-3-8B | Pending — not yet smoke-tested |
| Suite B | Multi-chip, Llama-3-70B | MCCL tensor parallelism |
| Suite C | Quantization, Llama-3.1-8B | FP8 skipped; GPTQ/AWQ; W8A8/W8A16 via compressed-tensors |
| Suite D | Long context ~28K input | Reduce `max_num_seqs` / `gpu_memory_utilization` in runner config |
| Suite E | Multi-chip scaling, Llama-3-8B | MCCL tensor parallelism |
| Suite F | Edge, Qwen2.5-0.5B | Pending |
| Suite G | MoE multi-chip, Mixtral-8x7B | Pending |
| Suite H | Mistral-7B SWA | Pending |

## Hardware compatibility

| GPU | BF16 | FP16 | Multi-chip TP | FP8 | Notes |
|-----|------|------|---------------|-----|-------|
| MetaX MXC series (N260 / C500) | ✅ (preferred) | ✅ | ✅ (MCCL) | ❌ | vllm-metax required |

FP8 / mxfp4 are **not** supported on MACA. FP32 inference is not consistently
supported — excluded. Marlin kernels are CUDA-only; use `gptq` (MACA kernel),
not `gptq_marlin`.

## Prerequisites

Install in this order — **do not** `pip install torch` or `vllm` from PyPI on
a bare Linux host:

1. **MXMACA SDK + driver** — from [developer.metax-tech.com](https://developer.metax-tech.com)
   (requires registration). Source the MACA environment (`MACA_PATH`, cu-bridge,
   etc.) before using the runner.
2. **vllm-metax** — the MetaX vLLM backend. Distributed as a Docker image or
   offline wheels; there is no public PyPI package.
3. **Runner dependencies**:

   ```bash
   pip install -r runners/metax_vllm_maca_34368fb0/requirements.txt
   ```

## Smoke test

```bash
python runners/metax_vllm_maca_34368fb0/test_smoke.py
python runners/metax_vllm_maca_34368fb0/test_smoke.py /path/to/model
```

## Usage

```bash
python run.py --runner metax_vllm_maca_34368fb0 --suite suite_A --precision BF16

# Multi-chip tensor parallelism (MCCL)
python run.py --runner metax_vllm_maca_34368fb0 \
  --suite suite_B --tensor-parallel-size 8
```

Optional runner config (copy and edit):

```bash
cp configs/runner_configs/runner_metax_vllm_maca_34368fb0.yaml.example \
   configs/runner_configs/runner_metax_vllm_maca_34368fb0.yaml
```

| Field | Default | Notes |
|-------|---------|-------|
| `tensor_parallel_size` | 1 | MCCL tensor parallelism |
| `enforce_eager` | false | Only if graph capture errors |
| `max_num_seqs` | 512 | Lower on small HBM |
| `gpu_memory_utilization` | 0.90 | Lower if OOM |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `torch.cuda` missing / no device | Source the MACA environment (`cu-bridge`) before running |
| OOM | Lower `gpu_memory_utilization` / `max_num_seqs` |
| Graph capture errors | `--enforce-eager` or `enforce_eager: true` in runner YAML |
| FP8 errors | FP8 unsupported — use BF16/FP16 |

## Requirements

See `requirements.txt` for AccelMark extras. vllm-metax and the MXMACA driver
are installed per the MetaX developer portal (not from this file).
