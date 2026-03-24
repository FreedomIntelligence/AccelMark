---
name: Community submission
about: Submit a benchmark result to the AccelMark leaderboard
title: "[Submission] <chip> — <suite> — <date>"
labels: submission
assignees: JuhaoLiang1997
---

<!--
Thank you for submitting to AccelMark!

Before opening this issue:
  ✓ Run: python runners/validate_submission.py --dir results/community/<your_dir>
  ✓ Validation must pass with no errors
  ✓ Attach result.json and env_info.json below (rename to .txt if GitHub blocks .json)

Title format: [Submission] NVIDIA A100-SXM4-80GB — suite_A — 2026-03-22
-->

## Hardware

| Field | Value |
|---|---|
| Chip name | <!-- e.g. NVIDIA A100-SXM4-80GB --> |
| Vendor | <!-- NVIDIA / AMD / Huawei / Apple / Other --> |
| Chip count | <!-- e.g. 1 --> |
| Memory per chip | <!-- e.g. 80 GB --> |
| Intra-node interconnect | <!-- e.g. NVLink 12 / PCIe Gen4 / N/A --> |
| Cloud provider & instance (if applicable) | <!-- e.g. AWS p4d.24xlarge --> |

## Software

| Field | Value |
|---|---|
| Framework | <!-- e.g. vLLM 0.6.6 --> |
| Driver version | <!-- e.g. 565.57.01 --> |
| CUDA / ROCm / runtime version | <!-- e.g. CUDA 12.1 --> |
| OS | <!-- e.g. Ubuntu 22.04 --> |
| Python | <!-- e.g. 3.10.18 --> |

## Benchmark

| Field | Value |
|---|---|
| Suite | <!-- suite_A / suite_B / suite_C / suite_D / suite_E --> |
| Model | <!-- e.g. meta-llama/Meta-Llama-3-8B-Instruct --> |
| Precision | <!-- BF16 / INT8 / INT4 --> |
| Runner | <!-- e.g. nvidia_vllm_3607f3ff --> |
| Run date | <!-- e.g. 2026-03-22 --> |

## Key results

<!-- Fill in the metrics most relevant to your suite -->

**Offline:** <!-- e.g. 5,321 tokens/sec -->
**Online max QPS:** <!-- e.g. 5 QPS -->
**Interactive TTFT p99:** <!-- e.g. 69 ms -->
**Accuracy valid:** <!-- Yes / No -->

## Validation output

```
# Paste the output of: python runners/validate_submission.py --dir results/community/<your_dir>
```

## Attached files

<!-- Attach result.json and env_info.json here. Rename to .txt if GitHub blocks the upload. -->

- [ ] `result.json` attached
- [ ] `env_info.json` attached (if available)

## Checklist

- [ ] I ran `validate_submission.py` and it passed with no errors
- [ ] Results are from an unmodified runner script (or I am submitting a new platform — see CONTRIBUTING.md)
- [ ] I ran the full number of runs specified by the suite (`num_runs: 3`)
- [ ] I am the person who ran this benchmark (no third-party results without their consent)
- [ ] If I am a vendor employee, I have tagged `[vendor]` in my submitter name in `result.json`
