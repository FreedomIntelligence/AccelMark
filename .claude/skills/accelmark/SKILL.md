---
name: accelmark
description: Run, validate, or query AccelMark LLM inference benchmarks across AI accelerators (NVIDIA, AMD, Huawei Ascend, Apple Silicon, Google TPU, Moore Threads). Use when a user wants to benchmark their GPU/NPU, check environment readiness, look up chip rankings or cost-efficiency, find wanted hardware gaps, or prepare a community PR submission.
---

# AccelMark Skill

Help the user either **(A) run and submit a real-world benchmark** on their own hardware, or **(B) query existing leaderboard data** (zero GPU required).

> **Core philosophy**: Out-of-the-box default configurations on consumer, older, or domestic accelerators are just as valuable as tuned SOTA runs. Never pressure the user to beat #1 before submitting.

## Mode A: Check Environment, Run Benchmark & Prepare Submission

1. **Pre-flight check (10 seconds, no model download)**:
   ```bash
   python run.py --check-env
   # Or check a specific runner:
   python run.py --check-env --runner nvidia_vllm_47f5d58e
   ```
   Verify Python >= 3.10, `jsonschema` / `numpy` / `pyyaml`, `configs/submitter.yaml`, and hardware visibility (`nvidia-smi`, `npu-smi`, `rocm-smi`, or Apple Silicon).

2. **Pick the fastest matching suite** (unless the user specifies one):
   - **Consumer / Edge / Quickest (~10 min)**: `suite_F` (`Qwen2.5-0.5B`, 4GB+ VRAM)
   - **Standard Single-Chip (~11 min)**: `suite_A` (`Llama-3-8B`, 16GB+ VRAM)
   - **Quantization Tradeoff (~22 min)**: `suite_C` (`Llama-3.1-8B` BF16/FP8/W8A8/W8A16/W4A16)
   - **Long Context 28K (~22 min)**: `suite_D` (`Llama-3.1-8B`)
   - **Sliding-Window Attention (~14 min)**: `suite_H` (`Mistral-7B`, 10K ctx, 4K window)
   - **Multi-Chip / Large / MoE**: `suite_B` (`Llama-3-70B`), `suite_E` (TP scaling 1x–8x), `suite_G` (`Mixtral-8x7B`)

3. **Run & validate**:
   ```bash
   python run.py --runner <runner_id> --suite <suite_id>
   python runners/validate_submission.py --dir results/community/<run_name>
   python tools/pr_rank_preview.py results/community/<run_name>
   ```

4. **Open a PR** (CLI or GitHub Web UI):
   - **CLI**: `git checkout -b submit/<hardware> && git add results/community/<run_name>/ && git commit -m "results: <hardware> on <suite>" && git push origin HEAD`
   - **No `gh` CLI / Web UI**: Upload `results/community/<run_name>/result.json` and `env_info.json` directly via GitHub's "Add file → Upload files" on a fork and open a Pull Request.

## Mode B: Zero-GPU Hardware & Engine Advisor

When the user has no GPU or wants to compare chips, engines, or cost-efficiency (`tok/s per $/hr`), run the built-in advisor against the local static API (`leaderboard/site/api/`):

```bash
# List top chips & cost-efficiency for a suite
python tools/advisor.py --suite suite_A

# Compare two or more chips side by side
python tools/advisor.py --compare "RTX 4090" "RTX 3090" "A100"

# Show open wanted hardware or reproduction quests
python tools/advisor.py --wanted
```
