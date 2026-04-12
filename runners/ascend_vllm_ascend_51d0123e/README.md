# ascend_vllm_ascend_6ebe6ef9 — Huawei Ascend NPU Runner

AccelMark runner for Huawei Ascend NPUs using [vllm-ascend](https://github.com/vllm-project/vllm-ascend), the vLLM community fork for CANN.

## Supported suites

| Suite | Description | Notes |
|-------|-------------|-------|
| Suite A | Single-chip, Llama-3-8B | Reference suite |
| Suite B | Multi-chip, Llama-3-70B | Requires multiple Ascend chips via HCCL TP |
| Suite C | Quantization, Llama-3.1-8B | FP8 skipped (not supported on 910B/910C); W8A8/W8A16 via compressed-tensors; W4A16 via gptq |
| Suite D | Long context ~28K input, Llama-3.1-8B | Reduce `max_num_seqs` and `gpu_memory_utilization` in runner config |
| Suite E | Multi-chip scaling, Llama-3-8B | HCCL-based tensor parallelism |
| Suite F | Consumer/edge, Qwen2.5-0.5B | Single-chip |

## Hardware compatibility

| NPU | BF16 | Multi-chip TP | FP8 | Notes |
|-----|------|---------------|-----|-------|
| Ascend 910B | ✅ | ✅ (HCCL) | ❌ | Full support for Suites A–F |
| Ascend 910C | ✅ | ✅ (HCCL) | ❌ | Full support for Suites A–F |

FP8 is excluded entirely — there is no native FP8 hardware support on current Ascend 910B/910C.
Suite C will skip the FP8 precision tier automatically and run BF16/W8A8/W8A16/W4A16 only.

## Prerequisites

Install in this order — Python packages alone are not sufficient:

**1. CANN toolkit**

Download from [https://www.hiascend.com/software/cann](https://www.hiascend.com/software/cann).
Match the CANN version to your NPU driver version exactly.

**2. torch_npu**

Huawei's PyTorch extension for Ascend NPU.
Install instructions: [https://gitee.com/ascend/pytorch](https://gitee.com/ascend/pytorch)

**3. Runner dependencies**

```bash
pip install -r runners/ascend_vllm_ascend_6ebe6ef9/requirements.txt
```

## Basic usage

```bash
# Run Suite A (standard datacenter benchmark)
python run.py --runner ascend_vllm_ascend_6ebe6ef9 --suite suite_A

# Run Suite F (edge/consumer benchmark)
python run.py --runner ascend_vllm_ascend_6ebe6ef9 --suite suite_F

# Run a single scenario
python run.py --runner ascend_vllm_ascend_6ebe6ef9 --suite suite_A --scenario offline

# Multi-chip run (Suite B, 4× Ascend 910B)
python run.py --runner ascend_vllm_ascend_6ebe6ef9 --suite suite_B --tensor-parallel-size 4

# Use a local model cache
python run.py --runner ascend_vllm_ascend_6ebe6ef9 --suite suite_A \
    --model-path /data/models/Meta-Llama-3-8B-Instruct
```

## Runner config

Copy the example config and adjust for your hardware:

```bash
cp configs/runner_configs/runner_ascend_vllm_ascend_6ebe6ef9.yaml.example \
   configs/runner_configs/runner_ascend_vllm_ascend_6ebe6ef9.yaml
```

Key settings:

| Field | Default | Notes |
|-------|---------|-------|
| `tensor_parallel_size` | 1 | Number of NPUs for tensor parallelism |
| `enforce_eager` | false | Disable CANN graph compilation; set true if graph errors occur |
| `max_num_seqs` | 512 | Max concurrent sequences; reduce on lower-memory NPUs |
| `gpu_memory_utilization` | 0.90 | Fraction of NPU HBM reserved for KV cache; reduce if OOM |

## CANN graph compilation errors

If you encounter errors during graph compilation (e.g. unsupported ops on your CANN version),
disable graph capture with `--enforce-eager`:

```bash
python run.py --runner ascend_vllm_ascend_6ebe6ef9 --suite suite_A --enforce-eager
```

Or set persistently in the runner config YAML:
```yaml
enforce_eager: true
```

## NPU OOM errors

If you encounter out-of-memory errors, reduce `gpu_memory_utilization` and/or `max_num_seqs`
in the runner config YAML, either globally or per-suite:

```yaml
gpu_memory_utilization: 0.85
max_num_seqs: 256

suites:
  suite_D:
    max_num_seqs: 64
    gpu_memory_utilization: 0.80
```

## Suite D notes

Suite D uses ~28K token input sequences. On NPUs with tighter HBM budgets,
reduce both `max_num_seqs` (to `64` or lower) and `gpu_memory_utilization`
(to `0.85` or lower) under the `suites.suite_D` override in your runner config.

## Requirements

See `requirements.txt` for the pinned dependency list.

Minimum environment:
- Huawei Ascend 910B or 910C NPU
- CANN toolkit (version matched to NPU driver)
- torch_npu (Huawei PyTorch NPU extension)
- Python 3.10+
- vllm-ascend ≥ 0.7.0, < 0.8.0
