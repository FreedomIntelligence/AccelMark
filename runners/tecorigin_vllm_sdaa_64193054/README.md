# tecorigin_vllm_sdaa_64193054 — TecoOrigin SDAA Runner (Teco-vLLM)

AccelMark runner for TecoOrigin SDAA accelerators using **Teco-vLLM**.

## Supported suites

| Suite | Description | Notes |
|-------|-------------|-------|
| Suite A | Single-chip, Llama-3-8B | Pending — not yet smoke-tested |
| Suite B | Multi-chip, Llama-3-70B | SDAA interconnect tensor parallelism |
| Suite C | Quantization, Llama-3.1-8B | FP8 skipped; GPTQ/INT8; W8A8/W8A16 via compressed-tensors |
| Suite D | Long context ~28K input | Reduce `max_num_seqs` / `gpu_memory_utilization` in runner config |
| Suite E | Multi-chip scaling, Llama-3-8B | SDAA interconnect |
| Suite F | Edge, Qwen2.5-0.5B | Pending |
| Suite G | MoE multi-chip, Mixtral-8x7B | Pending |
| Suite H | Mistral-7B SWA | Pending |

## Hardware compatibility

| Accelerator | BF16 | FP16 | Multi-chip TP | FP8 | Notes |
|-------------|------|------|---------------|-----|-------|
| TecoOrigin SDAA-200 / SDAA-400 | ✅ | ✅ | ✅ (SDAA interconnect) | ❌ | Teco-vLLM required |

FP8 is **not** supported on SDAA. Marlin kernels are CUDA-only; use `gptq`
(SDAA kernel), not `gptq_marlin`. Teco-vLLM presents SDAA as CUDA-compatible,
so no explicit `device=` flag is required.

## Prerequisites

Install in this order — **do not** `pip install torch` or `vllm` from PyPI on
a bare Linux host:

1. **SDAA SDK + torch-sdaa** — obtained from TecoOrigin (proprietary; no public PyPI).
2. **Teco-vLLM** — the vLLM adaptation for the SDAA architecture.
3. **Runner dependencies**:

   ```bash
   pip install -r runners/tecorigin_vllm_sdaa_64193054/requirements.txt
   ```

## Smoke test

```bash
python runners/tecorigin_vllm_sdaa_64193054/test_smoke.py
python runners/tecorigin_vllm_sdaa_64193054/test_smoke.py /path/to/model
```

## Usage

```bash
python run.py --runner tecorigin_vllm_sdaa_64193054 --suite suite_A --precision BF16

# Multi-chip tensor parallelism
python run.py --runner tecorigin_vllm_sdaa_64193054 \
  --suite suite_B --tensor-parallel-size 8
```

Optional runner config (copy and edit):

```bash
cp configs/runner_configs/runner_tecorigin_vllm_sdaa_64193054.yaml.example \
   configs/runner_configs/runner_tecorigin_vllm_sdaa_64193054.yaml
```

| Field | Default | Notes |
|-------|---------|-------|
| `tensor_parallel_size` | 1 | SDAA interconnect tensor parallelism |
| `enforce_eager` | false | Only if graph capture errors |
| `max_num_seqs` | 512 | Lower on small HBM |
| `gpu_memory_utilization` | 0.90 | Lower if OOM |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `torch.cuda` missing / no device | Ensure `torch-sdaa` is installed and loadable |
| OOM | Lower `gpu_memory_utilization` / `max_num_seqs` |
| Graph capture errors | `--enforce-eager` or `enforce_eager: true` in runner YAML |
| FP8 errors | FP8 unsupported — use BF16/FP16 |

## Requirements

See `requirements.txt` for AccelMark extras. The SDAA SDK, torch-sdaa, and
Teco-vLLM are installed per TecoOrigin's distribution (not from this file).
