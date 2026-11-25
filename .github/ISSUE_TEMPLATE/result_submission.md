---
name: Submit Benchmark Result (No-Git Upload)
about: Submit a benchmark result if you prefer not to open a Pull Request via git
title: "[Result] <Hardware> on <Suite>"
labels: ["result-submission"]
assignees: []
---

### Hardware & Suite
- **Accelerator**: <!-- e.g. NVIDIA RTX 4090 (24GB), Huawei Ascend 910B, Apple M4 Max -->
- **Chip count**: <!-- e.g. 1, 4, 8 -->
- **Suite**: <!-- e.g. suite_F, suite_A, suite_C -->
- **Runner**: <!-- e.g. nvidia_vllm_47f5d58e -->

### Attach Your Result Files
Drag and drop either:
- The `.zip` produced by `python run.py --submit results/community/<run_name> --zip`, **or**
- Both `result.json` and `env_info.json` from `results/community/<run_name>/`.

<!-- Tip: You can validate locally first with:
     python runners/validate_submission.py --dir results/community/<run_name> -->

### Local Validation Output (optional)
```text
<!-- Paste output of python run.py --submit results/community/<run_name> -->
```

### Notes (optional)
<!-- Any special flags (--enforce-eager, quantized weights, custom driver/framework version, etc.) -->
