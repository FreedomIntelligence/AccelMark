# AMD Platform Scripts

Benchmark scripts for AMD GPUs (MI300X) using vLLM with ROCm backend.

## Requirements

```bash
pip install -r scripts/amd/requirements.txt
```

ROCm 6.x and AMD GPU drivers required.

## vLLM ROCm (run_vllm_rocm.py)

Supports Suite A (offline, online, interactive).

```bash
python scripts/amd/run_vllm_rocm.py \
    --suite suite_A \
    --scenario offline \
    --output-dir ./my_submission/ \
    --tensor-parallel-size 1
```

## Notes

- vLLM supports ROCm via the same API as CUDA. The script is nearly identical to the NVIDIA vLLM script.
- Set `HIP_VISIBLE_DEVICES` to select specific GPUs.
- For MI300X with 192GB HBM, suite B (70B model) may fit on a single chip.
