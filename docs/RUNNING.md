# Running AccelMark Benchmarks

## Prerequisites

### 1. Clone the repo

```bash
git clone https://github.com/JuhaoLiang1997/AccelMark.git
cd AccelMark
```

### 2. Set up Python environment

```bash
# NVIDIA GPUs
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r scripts/nvidia/requirements.txt
```

### 3. Configure submitter profile

```bash
cp configs/submitter.yaml.example configs/submitter.yaml
# Edit configs/submitter.yaml with your GitHub username
```

### 4. Configure local model paths (optional)

```bash
cp configs/models.yaml configs/models_local.yaml
# Edit configs/models_local.yaml to add local paths for downloaded models
# This avoids re-downloading models from HuggingFace on every run
```

---

## Running a benchmark

### Single scenario

```bash
python scripts/nvidia/run_vllm.py \
    --suite suite_A \
    --scenario offline \
    --output-dir ./results/community/my_gpu_suite_A_offline
```

### All scenarios at once (recommended)

```bash
python scripts/nvidia/run_vllm.py \
    --suite suite_A \
    --scenario all \
    --output-dir ./results/community/my_gpu_suite_A
```

This runs offline → online → interactive in sequence and produces:
```
results/community/my_gpu_suite_A/
├── result.json          ← merged suite-level result
├── offline/result.json
├── online/result.json
└── interactive/result.json
```

### Multi-chip

```bash
python scripts/nvidia/run_vllm.py \
    --suite suite_A \
    --scenario all \
    --output-dir ./results/community/my_gpu_x2_suite_A \
    --tensor-parallel-size 2
```

---

## What gets measured

| Scenario | Primary metric | What it means |
|----------|---------------|---------------|
| offline | tokens/sec | Max throughput when GPU is fully loaded |
| online | max_valid_qps | Max requests/sec while keeping p99 TTFT < 500ms |
| interactive | TTFT p99 | Single-user latency with no queueing |

---

## Expected run times (A100 SXM4 80GB reference)

| Scenario | Time |
|----------|------|
| offline | ~5 min |
| online | ~10 min |
| interactive | ~12 min |
| **all** | **~27 min** |

---

## Troubleshooting

**enforce_eager error / torch compile error**
```bash
# Add --enforce-eager flag and note it in result.json
python scripts/nvidia/run_vllm.py --suite suite_A --scenario all \
    --output-dir ./results/... --enforce-eager
```

**OOM on Suite D**
OOM at some batch sizes is expected for long-context suite.
The benchmark records `"oom": true` and continues.

**Model not found**
Set `local_path` in `configs/models_local.yaml` or pass `--model-path`.
