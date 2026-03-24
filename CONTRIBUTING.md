# Contributing to AccelMark

AccelMark is a community-driven benchmark leaderboard for AI accelerators.
The most valuable contribution is running the benchmark on hardware not yet
in the leaderboard and submitting your results.

---

## Quick start

**Got a GPU? Here's the shortest path to getting on the leaderboard:**

```bash
# 1. Clone and install
git clone https://github.com/JuhaoLiang1997/AccelMark.git
cd AccelMark
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r runners/nvidia_vllm_3607f3ff/requirements.txt

# 2. Set your name (one-time setup)
cp configs/submitter.yaml.example configs/submitter.yaml
# Edit configs/submitter.yaml — add your name or GitHub username

# 3. Run the benchmark (~46 min on A100)
#    Accuracy gate runs automatically before the benchmark starts.
#    Output directory is auto-named:
#    results/community/nvidia_a100sxm480gbx1_suite_A_nvidia_vllm_3607f3ff
python run.py --runner nvidia_vllm_3607f3ff --suite suite_A

# 4. Submit
ls results/community/
python runners/validate_submission.py --dir results/community/<your_submission_dir>
# Then open a GitHub Issue using the "Community Submission" template
```

That's it. The CI bot handles the rest.

---

## One-time setup

### Install dependencies

**NVIDIA (vLLM):**
```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r runners/nvidia_vllm_3607f3ff/requirements.txt
```

**Other platforms:**
```bash
# List all available runners and their install instructions
python run.py --list
# Then: pip install -r runners/<runner_id>/requirements.txt
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
cp configs/models_local.yaml.example configs/models_local.yaml
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

## Running the benchmark

### Recommended: run the full suite

```bash
python run.py --runner nvidia_vllm_3607f3ff --suite suite_A

# Override the output directory if needed
python run.py --runner nvidia_vllm_3607f3ff \
    --suite suite_A \
    --output-dir ./results/verified/nvidia_a100sxm480gbx1_suite_A_nvidia_vllm_3607f3ff
```

This runs the suite's default scenarios (accuracy gate → offline → online → interactive)
in sequence and produces a single merged `result.json`. If the accuracy gate fails,
the benchmark is aborted (use `--skip-accuracy-gate` to override).

Use `--scenario all` to also include extra scenarios defined in the suite (e.g. `sustained`),
or `--scenario offline` (or any other scenario name) to run a single scenario.

### Run a single scenario

```bash
python run.py --runner nvidia_vllm_3607f3ff --suite suite_A --scenario offline
```

### Multi-chip (Suite B and above)

```bash
# Suite B: Llama-3-70B on 4 chips
python run.py --runner nvidia_vllm_3607f3ff \
    --suite suite_B \
    --tensor-parallel-size 4
```

Suite B does not require a specific chip count — use however many chips
your hardware needs to fit the 70B model. The result records the actual
chip count and the leaderboard groups results by chip count for fair comparison.

> **Suite A on multiple chips is not recommended.** Llama-3-8B fits on a single chip,
> so multi-chip adds communication overhead without a meaningful use case.
> Use Suite B (70B) or Suite E (scaling benchmark) for multi-chip runs.

### Suite E: multi-chip scaling

Suite E runs the same workload at 1×, 2×, 4×, and 8× chip counts and
measures how efficiently throughput scales:

```bash
python run.py --runner nvidia_vllm_3607f3ff \
    --suite suite_E \
    --max-chips 4
```

### With a local model path

```bash
python run.py --runner nvidia_vllm_3607f3ff \
    --suite suite_A \
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

### Accuracy gate (automatic)

When you run `--scenario all`, accuracy runs automatically as the **first step**
before any benchmark scenarios start. If accuracy fails, the benchmark is aborted.

```
============================================================
  Step 1: Accuracy Gate
  Must pass before benchmark runs.
============================================================

Score: 62/100 = 0.6200
Baseline: 0.6000
Delta: +0.0200 (min allowed: -0.03)
Valid: True

  ✓ Accuracy gate passed: 0.62 (delta=0.02, valid=True)
```

The accuracy check uses the **same model instance** as the benchmark — same
framework, same precision, same inference stack. This ensures the accuracy
result reflects exactly what the benchmark measured.

The result is saved to `accuracy/accuracy.json` inside the output directory and
injected into `result.json` automatically.

**If accuracy fails:**
```
  ✗ ACCURACY GATE FAILED
  Score:     0.45
  Delta:     0.1500
  Threshold: 0.03

  Fix model weights before submitting.
  To run anyway: --skip-accuracy-gate
```

The benchmark is aborted. Common causes:
- Wrong model revision (update `model_revision` in suite.json)
- Quantized weights with too much quality loss
- Model loaded with wrong precision

**`--skip-accuracy-gate`** — run benchmark even if accuracy fails:
```bash
python run.py --runner nvidia_vllm_3607f3ff \
    --suite suite_A \
    --skip-accuracy-gate
```

Results submitted with a failed accuracy gate are flagged on the leaderboard.
This flag is permanent — it cannot be removed by re-running. Only use
`--skip-accuracy-gate` for debugging or stress testing.

**Running accuracy standalone** (optional):

If you want to check accuracy before committing to a full benchmark run,
you can run accuracy as its own scenario:
```bash
python run.py --runner nvidia_vllm_3607f3ff --suite suite_A --scenario accuracy
```

**Per-question outputs** (`accuracy_outputs.jsonl`):

Every accuracy run writes `accuracy_outputs.jsonl` alongside `accuracy.json`.
Each line records one question — the model's raw output, extracted answer,
ground truth, and whether it was correct. Useful for validating answer
extraction or debugging low scores.

This file is gitignored and only needed locally. It is **not** required for
submission.

**Resuming an interrupted run:**

If a run is interrupted, re-running the same command resumes from where it
stopped. Completed steps are detected by the presence of their output files
and skipped automatically:

- Accuracy gate: skipped if `accuracy/accuracy.json` already exists
- Each scenario: skipped if `<scenario>/result.json` already exists

```
  [○] accuracy     -- SKIPPED (already done)
  [○] offline      -- SKIPPED (already done)
  [✓] online       -- SUCCESS
  [✓] interactive  -- SUCCESS
```

### Step 1: Validate

```bash
# Find your auto-generated output directory
ls results/community/

# Validate it (replace the directory name with yours)
python runners/validate_submission.py \
    --dir results/community/nvidia_a100sxm480gbx1_suite_A_nvidia_vllm_3607f3ff
```

The validator checks:
- All required fields are present in `result.json`
- Accuracy check passed
- `submitted_by` is not empty
- Throughput values are non-zero and not anomalously high

**Files required for submission** (the rest are gitignored and stay local):

```
<submission_dir>/
  result.json                  # merged suite result — required
  env_info.json                # hardware environment — required
  accuracy/
    accuracy.json              # accuracy gate result — required
  offline/
    result.json
  online/
    result.json
  interactive/
    result.json
```

`run.log`, `samples.jsonl`, and `accuracy_outputs.jsonl` are gitignored and
stay on your machine — they are not part of the submission.

Fix any errors before submitting. If validate exits with no errors, you're ready.

### Step 2: Open a GitHub Issue

Go to: **https://github.com/JuhaoLiang1997/AccelMark/issues/new**

Select template: **Community Submission**

Paste the contents of your `result.json` into the issue body.

**Submission directory naming** is handled automatically. The benchmark
generates a standardized name based on chip, suite, and runner:

```
results/community/nvidia_a100sxm480gbx1_suite_A_nvidia_vllm_3607f3ff
                  ^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^
                  chip (count + name)   suite   runner id
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
python run.py --runner nvidia_vllm_3607f3ff --suite suite_A --model-path /your/path/...
```

If your local copy was downloaded at a different time but has identical
weight files, add a note in `meta.notes`:
```json
"notes": "Local copy, weights identical to locked revision 8afb486c"
```

---

## Adding support for a new platform

Create a new runner folder under `runners/` by subclassing `BenchmarkRunner`.
See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the full implementation guide including
how to compute your runner's hash ID.

```python
# runners/your_platform_{hash8}/runner.py
from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
from loadgen.types import InferenceResult

class MyFrameworkRunner(BenchmarkRunner):

    SUPPORTS_STREAMING  = True    # set False if no streaming API
    SUPPORTS_BATCHING   = True    # set False if serial only (e.g. mlx-lm)
    SUPPORTS_ONLINE     = True
    SUPPORTS_MULTI_CHIP = True    # set False if no tensor parallelism

    def load_model(self, model_path: str, suite: dict, parallelism: dict) -> None:
        # Read only the parallelism keys you need — ignore others
        tp_size = parallelism["tensor_parallel_size"]
        self.model = MyFramework.load(model_path, tp=tp_size)

    def inference_fn_offline(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        prompts = [self._format_prompt(r.prompt) for r in requests]
        outputs = self.model.generate(prompts)
        return [InferenceResult(
            first_token_time_ms=None,
            total_time_ms=o.elapsed_ms,
            output_tokens=o.num_tokens,
            input_tokens=o.num_input_tokens,
            success=True,
        ) for o in outputs]

    async def inference_fn_streaming(self, request: InferenceRequest) -> InferenceResult:
        ...  # required if SUPPORTS_STREAMING = True

    def release_resources(self) -> None:
        del self.model
        self.model = None

    def _get_framework_name(self) -> str:
        return "MyFramework"

    def _get_framework_version(self) -> str:
        import myframework
        return myframework.__version__

if __name__ == "__main__":
    MyFrameworkRunner().main()
```

All orchestration (result building, accuracy, suite dispatch, etc.) is
inherited from `BenchmarkRunner` automatically.

See `runners/template/runner.py` for a fully annotated starting point.

**Checklist for a new runner PR:**
- [ ] Runner folder named `{platform}_{name}_{hash8}` with correct hash
- [ ] `runner.py` subclasses `BenchmarkRunner` and passes `runners/validate_runners.py`
- [ ] `meta.json` present and valid
- [ ] `requirements.txt` included
- [ ] At least one reference result in `results/community/`
- [ ] `runners/collect_env.py` detects your hardware

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the full implementation reference.

---

## Reporting a Suspicious Result

If you think a result looks wrong:

1. Open a GitHub Issue using the **"Challenge a Result"** template
2. Include: the submission name, what looks wrong, and ideally your own run
   on the same hardware as evidence

Be specific. Maintainers will investigate and may move the result to `flagged`.

---

## Other ways to contribute

- **Fix a bug** — open a PR, include a test if possible
- **Improve runners** — better error messages, edge case handling
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