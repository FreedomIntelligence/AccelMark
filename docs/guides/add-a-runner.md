# Add a New Runner or Accelerator Family

Every inference engine on AccelMark lives in a self-contained, content-hashed directory under `runners/{platform}_{framework}_{hash8}/`.

## 5 Steps to Add a Runner

1. **Subclass `BenchmarkRunner`** in `runners/<tmp>/runner.py` implementing `load_model()`, `generate()`, and `release_resources()`.
2. **Compute the content hash** (first 8 hex characters of SHA-256 of `runner.py`):
   ```bash
   python runners/hash_runner.py runners/<tmp>/runner.py
   # → e.g. nvidia_lmdeploy_3f8a2c1d
   ```
3. **Rename the folder** to match the hash and write `meta.json` (declare `runner_tier`: `"reference"`, `"optimized"`, or `"experimental"`, plus supported suites).
4. **Keep it clean (no vendoring)**: Import upstream packages via `requirements.txt` — CI rejects PRs that vendor upstream engine source trees.
5. **Validate locally** before opening your PR:
   ```bash
   python runners/validate_runners.py
   python run.py --runner <your_folder> --suite suite_F --dry-run
   ```

!!! note "Do not edit `README.md`"
    The README platforms matrix is auto-generated from `runners/*/meta.json`. Maintainers refresh it with `python tools/generate_platforms_matrix.py`.

For the full subclass API, precision resolution rules, and adding a brand-new hardware platform plugin (`collect_env.py`), see [`DEVELOPMENT.md`](https://github.com/FreedomIntelligence/AccelMark/blob/main/DEVELOPMENT.md) and [`runners/README.md`](https://github.com/FreedomIntelligence/AccelMark/blob/main/runners/README.md).
