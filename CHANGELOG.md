# Changelog

All notable changes to AccelMark are documented in this file.

## [0.2.0] - 2026-11-26

### Added
- **Unified onboarding & submission CLI (`run.py`)**: `--init` one-time setup, `--check-env` pre-flight check (with `HF_TOKEN`/`HF_ENDPOINT` hints), `--runner auto` and short aliases (`vllm`, `sglang`, `trtllm`, `rocm`, `ascend`, `mlx`, `tpu`, `musa`, `iluvatar`), and `--submit [dir] [--zip]` with local validation + projected rank preview.
- **Zero-GPU Advisor (`tools/advisor.py`)**: `--budget`, `--tier`, `--engine-compare`, `--compare`, `--wanted`, and `--format table|markdown|json` output modes with automatic index generation on fresh clones.
- **New hardware & community results**: Iluvatar CoreX MR-V100 platform & runner (Suites A–H), Moore Threads MTT S4000 multi-GPU scaling (Suites B & E), NVIDIA RTX 4090 D 48GB, and A100 SGLang v6 optimized runner.
- **Agent Track (`#/agent`)**: AccelMark-Agent 50-cell A100 matrix, cross-accelerator portability tracks, and protocol audit cases.
- **Protocol integrity & runner governance**: `runner_tier` (`reference` / `optimized` / `experimental`) separation in README, CLI, and Rankings filters; `check_protocol_integrity()` in `validate_submission.py`; FP16→BF16 accuracy baseline fallback; CI guard against committing `.vendor/` trees; OpenGraph/Twitter social preview metadata.
- **GitHub Issue templates**: `result_submission.md` (no-git upload), `challenge_result.md`, and `hardware_request.md`.

### Fixed
- Fork PR CI no longer fails with `403 Resource not accessible by integration` when posting validation comments (writes to `$GITHUB_STEP_SUMMARY` and degrades comment posting gracefully).
- `validate_submission.py` now includes `dies_per_card` in `run_id` verification (with backward-compatible legacy fallback) and accepts both `--dir` and positional paths.
- `runners/platforms/moorethreads.py` no longer claims non-MTT `torch.cuda` shim devices (such as Iluvatar GPUs).

## [0.1.0] - 2026-05-19

- Initial public release with Suites A–G, reference runners (NVIDIA, AMD, Huawei Ascend, Apple Silicon, Google TPU, Moore Threads), shared LoadGen, accuracy gate, and GitHub Pages leaderboard.
