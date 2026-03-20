# AccelMark

Open benchmark framework for evaluating AI accelerators on LLM workloads.

**Why AccelMark?**
- MLPerf is powerful but complex to contribute to and slow to update
- Vendor whitepapers are not comparable with each other
- No existing benchmark covers non-NVIDIA chips, MoE models, and long-context together

AccelMark defines a strict result schema and shared LoadGen component.
Each platform implements its own inference backend. Anyone with a GPU can contribute.

## Live Leaderboard

[accelmark.github.io/accelmark](https://accelmark.github.io/accelmark)

## Quickstart — Run Suite A on NVIDIA

```bash
git clone https://github.com/accelmark/accelmark
cd accelmark

# Step 1: collect environment info
python scripts/collect_env.py --output my_submission/env_info.json

# Step 2: install dependencies
pip install -r scripts/nvidia/requirements.txt

# Step 3: run benchmark
python scripts/nvidia/run_vllm.py \
  --suite suite_A \
  --scenario offline \
  --output-dir my_submission/

# Step 4: run accuracy check
python scripts/run_accuracy.py \
  --suite suite_A \
  --script scripts/nvidia/run_vllm.py \
  --output my_submission/accuracy.json

# Step 5: validate before submitting
python scripts/validate_submission.py --dir my_submission/

# Step 6: submit PR
# Copy my_submission/ to results/community/{name}/
# Open a pull request
```

## Suites

| Suite | Model | Chips | Scenarios | Primary Metric |
|-------|-------|-------|-----------|----------------|
| A | Llama-3-8B | 1 | offline, online, interactive | throughput / max valid QPS |
| B | Llama-3-70B | 8 | offline, online | throughput / max valid QPS |
| C | Llama-3-8B | 8 | training | tokens/sec |
| D | Llama-3-8B | 1 | offline, interactive | throughput (long context) |

## Supported Platforms

| Platform | Framework | Suite A | Suite B | Suite C | Suite D |
|----------|-----------|---------|---------|---------|---------|
| NVIDIA (H100/A100) | vLLM | ✓ | ✓ | — | ✓ |
| NVIDIA (H100/A100) | TensorRT-LLM | ✓ | — | — | — |
| NVIDIA | torchtitan | — | — | ✓ | — |
| AMD (MI300X) | vLLM ROCm | ✓ | — | — | — |
| Huawei Ascend | MindIE | ✓ | — | — | — |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
