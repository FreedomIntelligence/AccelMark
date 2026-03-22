# Contributing to AccelMark

AccelMark is a community-driven benchmark leaderboard for AI accelerators.
The most valuable contribution is running the benchmark on hardware not yet
in the leaderboard and submitting your results.

---

## Quick Start

**Got a GPU? Here's the shortest path to getting on the leaderboard:**

```bash
# 1. Clone and install
git clone https://github.com/JuhaoLiang1997/AccelMark.git
cd AccelMark
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r scripts/nvidia/requirements.txt

# 2. Set your name (one-time setup)
cp configs/submitter.yaml.example configs/submitter.yaml
# Edit configs/submitter.yaml — add your name or GitHub username

# 3. Run accuracy check (one-time per model, ~5-10 min)
#    Result is auto-saved to results/accuracy/ and reused on future runs
python scripts/run_accuracy.py \
    --model-path /path/to/Meta-Llama-3-8B-Instruct \
    --suite suite_A
# Saved to: results/accuracy/meta-llama-3-8b-instruct_BF16_2026-03-22.json
# Expected: Score ~0.62, Valid: True

# 4. Run the benchmark (~27 min on A100)
#    Output directory is auto-named: results/community/a100x1_llama3-8b_suite-A_YYYY-MM-DD
#    Accuracy is reused automatically on future runs
python scripts/nvidia/run_vllm.py --suite suite_A --scenario all

# 5. Submit
# Find your auto-generated directory name
ls results/community/
python scripts/validate_submission.py --dir results/community/<your_submission_dir>
# Then open a GitHub Issue using the "Community Submission" template
```

That's it. The CI bot handles the rest.

---

## One-time Setup

### Install dependencies

**NVIDIA (vLLM):**
```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r scripts/nvidia/requirements.txt
```

**AMD (ROCm):**
```bash
pip install -r scripts/amd/requirements.txt
```

**Apple Silicon (mlx-lm):**
```bash
pip install -r scripts/apple/requirements.txt
```

### Set your submitter profile

```bash
cp configs/submitter.yaml.example configs/submitter.yaml
```

Edit `configs/submitter.yaml`:
```yaml
# Shown publicly on the leaderboard — can be your name, GitHub username, or organization
submitted_by: your_name_or_github_username
submission_type: individual   # individual / vendor / research
organization: null            # optional, e.g. "MIT" or "Google DeepMind"
```

This file is gitignored — it never gets committed.

### Configure local model paths (optional)

If you have models downloaded locally, set their paths so you don't
need to specify `--model-path` every time:

```bash
cp configs/models.yaml configs/models_local.yaml
```

Edit `configs/models_local.yaml`:
```yaml
models:
  meta-llama/Meta-Llama-3-8B-Instruct:
    local_path: /your/path/to/Meta-Llama-3-8B-Instruct
  meta-llama/Llama-3.1-8B-Instruct:
    local_path: null   # not downloaded yet
```

`configs/models_local.yaml` is gitignored. Once configured, you don't
need `--model-path` on the command line — the benchmark script reads
the local path automatically.

---

## Running the Benchmark

### Recommended: run all scenarios at once

```bash
# Output dir is auto-generated — no need to specify it manually
python scripts/nvidia/run_vllm.py --suite suite_A --scenario all

# Override the output directory if needed (e.g. for re-runs or verified submissions)
python scripts/nvidia/run_vllm.py \
    --suite suite_A \
    --scenario all \
    --output-dir ./results/verified/a100x1_llama3-8b_suite-A_2026-03-22
```

This runs offline → online → interactive in sequence (~27 min on A100)
and produces a single merged `result.json` for leaderboard submission.

### Run a single scenario

```bash
python scripts/nvidia/run_vllm.py --suite suite_A --scenario offline
```

### Multi-chip

```bash
python scripts/nvidia/run_vllm.py \
    --suite suite_A \
    --scenario all \
    --tensor-parallel-size 2
```

### With a local model path

```bash
python scripts/nvidia/run_vllm.py \
    --suite suite_A \
    --scenario all \
    --model-path /path/to/local/model
```

---

## What Gets Measured

| Scenario | Primary metric | What it tells you |
|----------|---------------|-------------------|
| **offline** | tokens/sec | Max throughput when the GPU is fully loaded |
| **online** | max valid QPS | How many users/sec this chip can serve within a 500ms latency SLA |
| **interactive** | TTFT p99 | Single-user latency when the system is idle |

All three together give a complete picture of a chip's inference capability.

### Expected run times (A100 SXM4 80GB reference)

| Scenario | Time |
|----------|------|
| offline | ~5 min |
| online | ~19 min |
| interactive | ~22 min |
| **all (recommended)** | **~46 min** |

Faster hardware completes proportionally quicker. Slower hardware takes longer.

---

## Submitting Your Results

### Step 0: Run accuracy check (first time only)

The accuracy check verifies your model weights produce correct outputs.
Run it **once per model + precision combination** — the result is saved
to `results/accuracy/` and reused automatically on all future runs.

```bash
python scripts/run_accuracy.py \
    --model-path /path/to/model \
    --suite suite_A
```

The result is auto-saved to `results/accuracy/` with a standardized name:
```
results/accuracy/meta-llama-3-8b-instruct_BF16_2026-03-22.json
```

Expected output for Suite A (Llama-3-8B BF16):
```
Score: ~0.62   (baseline: 0.62, threshold: ±0.03)
Valid: True
Saved to: results/accuracy/meta-llama-3-8b-instruct_BF16_2026-03-22.json
```

On subsequent runs the benchmark script finds this file automatically:
```
Reusing accuracy from: results/accuracy/meta-llama-3-8b-instruct_BF16_2026-03-22.json
```

If your score falls outside the ±0.03 threshold, your submission will be
flagged. This usually means your model weights differ from the locked revision.

### Step 1: Validate

```bash
# Find your auto-generated output directory
ls results/community/

# Validate it (replace the directory name with yours)
python scripts/validate_submission.py \
    --dir results/community/a100x1_llama3-8b_suite-A_2026-03-22
```

The validator checks:
- All required fields are present in `result.json`
- Accuracy check passed
- `submitted_by` is not empty
- Throughput values are non-zero and not anomalously high

Fix any errors before submitting. If validate exits with no errors, you're ready.

### Step 2: Open a GitHub Issue

Go to: **https://github.com/JuhaoLiang1997/AccelMark/issues/new**

Select template: **Community Submission**

Paste the contents of your `result.json` into the issue body.

**Submission directory naming** is handled automatically. The benchmark script
generates a standardized name based on your hardware, model, suite, and date:

```
results/community/a100x1_llama3-8b_suite-A_2026-03-22
                  ^^^^^ ^^^^^^^^^^^ ^^^^^^^ ^^^^^^^^^^
                  chip  model       suite   date
```

You can override this with `--output-dir` if needed.

The CI bot will automatically:
1. Validate the submission
2. Create a PR adding your result to `results/community/`
3. Update the leaderboard (usually within a few minutes)

### Step 3: Done

Your result appears on the **Community** tab of the leaderboard immediately
after the CI bot merges the PR. No manual review needed for community tier.

---

## Leaderboard Tiers

| Tier | How to get it | Leaderboard placement |
|------|--------------|----------------------|
| **community** | Submit via GitHub Issue, passes CI validation | Community tab |
| **verified** | Maintainer reproduced your result within 5% | Main leaderboard |

Most results live in community tier — this is completely normal.
Community results are visible and fully comparable; they just haven't
been independently reproduced yet.

To request verification, comment on your submission PR.

---

## Using Local or Air-gapped Models

AccelMark separates the **model identifier** (used for leaderboard comparisons)
from the **model path** (where weights are loaded from at runtime).

The `model_id` and `model_revision` in `result.json` are always the canonical
HuggingFace identifiers — they don't change regardless of where you load from.

**To download a model for offline use:**
```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir /your/path/Meta-Llama-3-8B-Instruct
```

**To use a local copy:**
```bash
# Option A: set it in configs/models_local.yaml (recommended)
# Option B: pass --model-path at runtime
python scripts/nvidia/run_vllm.py --model-path /your/path/...
```

If your local copy was downloaded at a different time but has identical
weight files, add a note in `meta.notes`:
```json
"notes": "Local copy, weights identical to locked revision 8afb486c"
```

---

## Adding Support for a New Platform

If your hardware isn't NVIDIA (e.g. AMD, Ascend, Apple Silicon), you can
add a new platform script.

1. Copy the template:
   ```bash
   cp scripts/template/run_benchmark.py scripts/your_platform/run_your_framework.py
   ```

2. Implement `inference_fn` for your platform. The function signature depends on scenario:
   ```python
   # For offline scenario (sync):
   def inference_fn(prompts: list[str]) -> list[InferenceResult]

   # For online and interactive scenarios (async):
   async def inference_fn(prompt: str) -> InferenceResult
   ```

3. Add `scripts/your_platform/requirements.txt`

4. Add `scripts/your_platform/README.md` with setup instructions

5. Submit a result using your new script as proof it works

The core `loadgen/loadgen.py` handles all timing and measurement logic —
your script only needs to implement inference. Do not write your own timing code.

---

## Reporting a Suspicious Result

If you think a result looks wrong:

1. Open a GitHub Issue using the **"Challenge a Result"** template
2. Include: the submission name, what looks wrong, and ideally your own run
   on the same hardware as evidence

Be specific. Maintainers will investigate and may move the result to `flagged`.

---

## Other Ways to Contribute

- **Fix a bug** — open a PR, include a test if possible
- **Improve platform scripts** — better error messages, edge case handling
- **Update cloud pricing** — edit `schema/cloud_pricing.json`, open a PR
  titled `data: update cloud pricing YYYY-MM` with source URLs
- **Propose a new suite** — open an Issue with model, chip count, scenarios,
  and rationale. New suites require a reference result before going live.

---

## A Few Rules

- Do not modify `schema/accuracy_subset.jsonl` — it is immutable
- Do not modify other people's results in `results/`
- Vendor employees may submit results for their own chips (shown with a Vendor badge)
- Results submitted with `--enforce-eager` are valid but noted — they may
  underrepresent true hardware capability