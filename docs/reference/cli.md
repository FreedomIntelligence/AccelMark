# CLI Reference

## `run.py` — Unified Benchmark Launcher

| Command | Purpose |
|---|---|
| `python run.py --init` | Interactive one-time submitter handle setup & 10-second pre-flight hardware/runner check (no model download) |
| `python run.py --list-runners` | List all installed runners with tier (`reference` / `optimized` / `experimental`) and supported suites |
| `python run.py --runner auto --suite suite_F` | Auto-detect hardware and run the default reference runner on Suite F |
| `python run.py --runner <alias_or_id> --suite <suite_id>` | Run a specific runner (aliases: `vllm`, `sglang`, `trtllm`, `rocm`, `ascend`, `iluvatar`, `musa`, `tpu`, `mlx`) |
| `python run.py --runner auto --suite suite_B --tensor-parallel-size 4` | Pass multi-GPU tensor parallelism flags through to the runner |
| `python run.py --submit results/community/<dir>` | Validate schema & protocol integrity, preview projected rank, and print exact PR commands |
| `python run.py --submit results/community/<dir> --zip` | Same validation + package a self-contained `.zip` for no-git GitHub Issue upload |

## `tools/advisor.py` — Zero-GPU Procurement & Engine Query Tool

| Command | Purpose |
|---|---|
| `python tools/advisor.py --suite suite_A --budget 2.0 --tier reference` | Rank Closed reference recipes under \$2.00/hr by tokens per dollar |
| `python tools/advisor.py --engine-compare "A100"` | Compare all inference engines (`vLLM`, `SGLang`, `TensorRT-LLM`, etc.) on the same accelerator |
| `python tools/advisor.py --compare "5090" "A100" "H100" --format markdown` | Export a Markdown or JSON comparison table |

## Validation & Maintenance Scripts

| Command | Purpose |
|---|---|
| `python runners/validate_submission.py results/community/<dir>` | Run exact PR CI checks locally (schema, `run_id` integrity, sanity bounds) |
| `python runners/validate_runners.py` | Verify all runner SHA-256 folder suffixes & `meta.json` schemas |
| `python tools/generate_platforms_matrix.py --check` | Check that the README platforms matrix matches `runners/*/meta.json` |
| `python tools/generate_visuals.py` | Regenerate `rank-shift-*.svg`, `roofline-suites-*.svg`, and `og-image.png` from live leaderboard data |
