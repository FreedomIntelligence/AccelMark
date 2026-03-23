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
pip install -r runners/nvidia_vllm_e0859b3c/requirements.txt

# 2. Set your name (one-time setup)
cp configs/submitter.yaml.example configs/submitter.yaml
# Edit configs/submitter.yaml — add your name or GitHub username

# 3. Run the benchmark (~46 min on A100)
#    Accuracy gate runs automatically before the benchmark starts.
#    Output directory is auto-named: results/community/nvidia_a100sxm480gbx1_suite_A_nvidia_vllm_e0859b3c
python run.py --runner nvidia_vllm_e0859b3c --suite suite_A --scenario all

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
pip install -r runners/nvidia_vllm_e0859b3c/requirements.txt
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
need `--model-path` on the command line.

---

## Running the benchmark

### Recommended: run all scenarios at once

```bash
python run.py --runner nvidia_vllm_e0859b3c --suite suite_A --scenario all
```

This runs the accuracy gate first, then offline → online → interactive in sequence
and produces a single merged `result.json`. If the accuracy gate fails, the benchmark
is aborted (use `--skip-accuracy-gate` to override).

```bash
# Override the output directory if needed
python run.py --runner nvidia_vllm_e0859b3c \
    --suite suite_A \
    --scenario all \
    --output-dir ./results/verified/nvidia_a100sxm480gbx1_suite_A_nvidia_vllm_e0859b3c
```

### Run a single scenario

```bash
python run.py --runner nvidia_vllm_e0859b3c --suite suite_A --scenario offline
```

### Multi-chip (Suite B and above)

```bash
# Suite B: Llama-3-70B on 4 chips
python run.py --runner nvidia_vllm_e0859b3c \
    --suite suite_B \
    --scenario all \
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
python run.py --runner nvidia_vllm_e0859b3c \
    --suite suite_E \
    --chip_counts 1,2,4
```

**Note on `concurrency` vs batch size:** The offline scenario sweeps
*client-side concurrency* (how many requests the load generator fires
simultaneously) — not the inference engine's internal batch size.
The engine's internal batching (e.g. vLLM's `max_num_seqs`) is
configured separately in the runner and is not varied by the suite.
Results report `concurrency` values, not batch sizes.

### With a local model path

```bash
python run.py --runner nvidia_vllm_e0859b3c \
    --suite suite_A \
    --scenario all \
    --model-path /path/to/local/model
```

---

## What gets measured

| Scenario | Primary metric | What it tells you |
|----------|---------------|-------------------|
| **offline** | tokens/sec | Max throughput when the GPU is fully loaded |
| **online** | max valid QPS | How many users/sec this chip can serve within a 500ms latency SLA |
| **interactive** | TTFT p99 | Single-user latency when the system is idle |

All three together give a complete picture of a chip's inference capability.

### Expected run times (A100-SXM4-80GB reference)

| Scenario | Time |
|----------|------|
| offline | ~5 min |
| online | ~19 min |
| interactive | ~22 min |
| **all (recommended)** | **~46 min** |

---

## Submitting your results

### Accuracy gate (automatic)

When you run `--scenario all`, accuracy runs automatically as the **first step**.
If accuracy fails, the benchmark is aborted.

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
framework, same precision, same inference stack.

**If accuracy fails:**
- Check model weights and revision against the suite spec
- Common cause: quantized weights with too much quality loss
- Use `--skip-accuracy-gate` only for debugging — results submitted with a failed
  accuracy gate are permanently flagged on the leaderboard

**Resuming an interrupted run:** Re-running the same command resumes from
where it stopped. Completed steps are skipped automatically.

### Step 1: Validate

```bash
ls results/community/
python runners/validate_submission.py \
    --dir results/community/nvidia_a100sxm480gbx1_suite_A_nvidia_vllm_e0859b3c
```

**Files required for submission:**

```
<submission_dir>/
  result.json          # merged suite result — required
  env_info.json        # hardware environment — required
  accuracy/
    accuracy.json      # accuracy gate result — required
  offline/result.json
  online/result.json
  interactive/result.json
```

`run.log`, `samples.jsonl`, and `accuracy_outputs.jsonl` are gitignored
and stay on your machine — they are not part of the submission.

### Step 2: Open a GitHub Issue

Go to [Issues → New → Community Submission](https://github.com/JuhaoLiang1997/AccelMark/issues/new?template=community_submission.md).

Paste the contents of your `result.json` into the issue body and attach
`env_info.json`. The CI bot will validate, create a PR, and update the
leaderboard automatically.

### Step 3: Done

Your result appears on the **Community** tab immediately after CI merges the PR.

---

## Leaderboard tiers

| Tier | How to get it | Leaderboard placement |
|------|--------------|----------------------|
| **community** | Submit via GitHub Issue, passes CI validation | Community tab |
| **verified** | Maintainer reproduced your result within 5% | Main leaderboard |

To request verification, comment on your submission issue.

---

## Using local or air-gapped models

AccelMark separates the **model identifier** (used for leaderboard comparisons)
from the **model path** (where weights load from at runtime).

`model_id` and `model_revision` in `result.json` always use canonical
HuggingFace identifiers — they don't change regardless of where you load from.

```bash
# Download for offline use
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir /your/path/Meta-Llama-3-8B-Instruct

# Use a local copy
python run.py --runner nvidia_vllm_e0859b3c --model-path /your/path/...
```

If your local copy was downloaded at a different revision, add a note in
`meta.notes` of `result.json`.

---

## Adding support for a new platform

Create a new runner folder under `runners/` by subclassing `BenchmarkRunner`.
See [docs/DEVELOPMENT.md](DEVELOPMENT.md) for the full implementation guide including
how to compute your runner's hash ID.

```python
# runners/your_platform_{hash8}/runner.py
from runners.benchmark_runner import BenchmarkRunner
from loadgen.types import InferenceResult

class MyFrameworkRunner(BenchmarkRunner):

    SUPPORTS_STREAMING = True
    SUPPORTS_BATCHING  = True
    SUPPORTS_ONLINE    = True
    SUPPORTS_MULTI_CHIP = True

    def load_model(self, model_path: str, suite: dict, tp_size: int) -> None:
        self.model = MyFramework.load(model_path, tp=tp_size)

    def inference_fn_offline(self, prompts: list[str]) -> list[InferenceResult]:
        outputs = self.model.generate(prompts)
        return [InferenceResult(
            first_token_time_ms=None,
            total_time_ms=o.elapsed_ms,
            output_tokens=o.num_tokens,
            input_tokens=o.num_input_tokens,
            success=True,
        ) for o in outputs]

    async def inference_fn_streaming(self, prompt: str) -> InferenceResult:
        ...  # required if SUPPORTS_STREAMING = True

    def release_resources(self) -> None:
        del self.model

    def _get_framework_name(self) -> str:
        return "MyFramework"

    def _get_framework_version(self) -> str:
        import myframework
        return myframework.__version__

if __name__ == "__main__":
    MyFrameworkRunner().main()
```

All orchestration (result building, accuracy reuse, Suite E, etc.) is
inherited from `BenchmarkRunner` automatically.

**Checklist for a new platform PR:**
- [ ] Runner folder named `{platform}_{name}_{hash8}` with correct hash
- [ ] `runner.py` subclasses `BenchmarkRunner` and passes `runners/validate_submission.py`
- [ ] `meta.json` present and valid (see `runners/meta.schema.json`)
- [ ] `requirements.txt` included
- [ ] At least one reference result in `results/community/`
- [ ] `runners/collect_env.py` updated to detect your hardware (see [DEVELOPMENT.md](DEVELOPMENT.md))
- [ ] `README.md` supported platforms table updated

See [docs/DEVELOPMENT.md](DEVELOPMENT.md) for the full implementation reference.

---

## Reporting a suspicious result

If a result looks wrong:

1. Open a GitHub Issue using the **"Challenge a Result"** template
2. Include: the submission name, what looks wrong, and ideally your own
   run on the same hardware as evidence

Maintainers will investigate and may move the result to `flagged/`.

---

## Other ways to contribute

- **Fix a bug** — open a PR with a description and test if possible
- **Improve runners** — better error messages, edge case handling
- **Update cloud pricing** — edit `schema/cloud_pricing.json`, open a PR
  titled `data: update cloud pricing YYYY-MM` with source URLs in the description
- **Propose a new suite** — open an Issue with model, chip count, scenarios,
  and rationale; new suites require a reference result before going live

---

## A few rules

- Do not modify `schema/accuracy_subset.jsonl` — it is immutable
- Do not modify other people's results in `results/`
- Vendor employees may submit results for their own chips (shown with a Vendor badge);
  disclose affiliation by tagging `[vendor]` in your submitter name
- Results submitted with `--enforce-eager` are valid but noted — they may
  underrepresent true hardware capability
- Results submitted with `--skip-accuracy-gate` are permanently flagged
