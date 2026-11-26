# First Submission (~10 Minutes)

> **Audience**: Anyone with a single GPU, NPU, TPU, or Mac.
> **Estimated time**: ~10 minutes on Suite F (`Qwen2.5-0.5B`). No tuning required.

!!! tip "No SOTA tuning required"
    Honest out-of-the-box reference baselines (`Closed` division) on consumer, older, or emerging accelerators are just as valuable as tuned datacenter runs.

## Step 1 · Pre-Flight Check

```bash
python run.py --init
```

Expected output:

```text
✓ Submitter profile saved to configs/submitter.yaml (@your-handle)
✓ Detected accelerator: NVIDIA GeForce RTX 4090 (24 GB)
✓ Default runner: nvidia_vllm_47f5d58e (reference)
```

## Step 2 · Run Suite F (~10 min)

```bash
python run.py --runner auto --suite suite_F
```

`--runner auto` picks the default reference runner for your detected accelerator (or pass aliases: `vllm`, `sglang`, `rocm`, `ascend`, `iluvatar`, `musa`, `tpu`, `mlx`).

What happens automatically:

1. **MMLU Accuracy Gate** runs first on the loaded model instance (`Δ ≤ 0.03` vs baseline).
2. **Offline + Online + Interactive** benchmark scenarios execute under the pinned LoadGen seed.
3. Results and hardware fingerprints are saved to `results/community/<run_dir>/` (`result.json` + `env_info.json`).

!!! note "Pre-Ampere NVIDIA GPUs (V100, T4, RTX 20xx)"
    Add `--enforce-eager` if Triton/CUDA-graph compilation fails on older architectures.

## Step 3 · Validate & Preview Your Rank

```bash
python run.py --submit results/community/<run_dir>
```

Expected output:

```text
✓ Schema validation passed
✓ Protocol & run_id integrity verified
✓ Projected rank on Suite F (Closed): #9 / 34
Next steps:
  git checkout -b submit/<run_dir>
  git add results/community/<run_dir>
  git commit -m "results: add <chip> suite_F"
  git push -u origin submit/<run_dir>
```

Prefer not to use `git`? Pass `--zip` instead and upload on GitHub Issues — see [Submit Without Git](../guides/submit-without-git.md).
