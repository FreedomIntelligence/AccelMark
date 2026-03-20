# NVIDIA Platform Scripts

Benchmark scripts for NVIDIA GPUs (H100, A100) using vLLM and TensorRT-LLM.

## Requirements

```bash
pip install -r scripts/nvidia/requirements.txt
```

CUDA 12.x and appropriate NVIDIA drivers required.

## vLLM (run_vllm.py)

Supports Suite A (offline, online, interactive), Suite B, Suite D.

```bash
python scripts/nvidia/run_vllm.py \
    --suite suite_A \
    --scenario offline \
    --output-dir ./my_submission/ \
    --tensor-parallel-size 1
```

For Suite B (8-chip):
```bash
python scripts/nvidia/run_vllm.py \
    --suite suite_B \
    --scenario offline \
    --output-dir ./my_submission/ \
    --tensor-parallel-size 8
```

## TensorRT-LLM (run_trtllm.py)

Supports Suite A only. Requires TensorRT-LLM installation and model compilation.

```bash
python scripts/nvidia/run_trtllm.py \
    --suite suite_A \
    --scenario offline \
    --output-dir ./my_submission/ \
    --engine-dir /path/to/compiled/engine/
```
