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
pip install -r runners/nvidia_vllm_47f5d58e/requirements.txt

# 2. Set your name (one-time setup)
cp configs/submitter.yaml.example configs/submitter.yaml
# Edit configs/submitter.yaml — add your name or GitHub username

# 3. Run the benchmark (~11 min on A100 for default scenarios)
#    Accuracy gate runs automatically before the benchmark starts.
#    Output directory is auto-named using run_name, e.g.:
    #    results/community/nvidia_a100_sxm4_80gbx1_suite_A_nvidia_vllm_47f5d58e_ed4b0557
    python run.py --runner nvidia_vllm_47f5d58e --suite suite_A

# 4. Submit — open a GitHub Issue and paste your result.json
# https://github.com/JuhaoLiang1997/AccelMark/issues/new?template=community_submission.md
```

That's it. The CI bot handles the rest.

---

## One-time setup

### Install dependencies

Each runner ships its own `requirements.txt`. Install for the runner you want to use:

```bash
pip install -r runners/nvidia_vllm_47f5d58e/requirements.txt
```

To see all available runners and their install commands:

```bash
python run.py --list
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
need `--model-path` on the command line.

---

## Running the benchmark

### Recommended: run the full suite

```bash
python run.py --runner nvidia_vllm_47f5d58e --suite suite_A
```

This runs the suite's default scenarios (accuracy gate → offline → online)
in sequence and produces a single merged `result.json`. If the accuracy gate fails,
the benchmark is aborted (use `--skip-accuracy-gate` to override).

Use `--scenario all` to also include extra scenarios defined in the suite (e.g. `interactive`, `sustained`),
or `--scenario offline` (or any other scenario name) to run a single scenario.

```bash
# Override the output directory if needed
python run.py --runner nvidia_vllm_47f5d58e \
    --suite suite_A \
    --output-dir ./results/verified/nvidia_a100_sxm4_80gbx1_suite_A_nvidia_vllm_47f5d58e_ed4b0557
```

### Run a single scenario

```bash
python run.py --runner nvidia_vllm_47f5d58e --suite suite_A --scenario offline
```

### Multi-chip (Suite B and above)

```bash
# Suite B: Llama-3-70B on 4 chips
python run.py --runner nvidia_vllm_47f5d58e \
    --suite suite_B \
    --tensor-parallel-size 4  # (or set tensor_parallel_size: 4 in the runner config yaml)
```

Suite B does not require a specific chip count — use however many chips
your hardware needs to fit the 70B model. The result records the actual
chip count and the leaderboard groups results by chip count for fair comparison.

> **Suite A on multiple chips is not recommended.** Llama-3-8B fits on a single chip,
> so multi-chip adds communication overhead without a meaningful use case.
> Use Suite B (70B) or Suite E (scaling benchmark) for multi-chip runs.

### Suite E: multi-chip scaling

Suite E runs the same workload at 1×, 2×, 4×, and 8× chip counts automatically
and measures how efficiently throughput scales. Specify the maximum chip count
you want to test:

```bash
python run.py --runner nvidia_vllm_47f5d58e \
    --suite suite_E
```

Set the chip count range via `tensor_parallel_size` in the runner config yaml,
or pass `--tensor-parallel-size` directly to the runner.

Suite E handles the scaling automatically — it runs at 1×, 2×, and 4× chips
sequentially and computes `scaling_efficiency = N_chip_throughput / (1_chip_throughput × N)`.

### Suite F: consumer / edge

Suite F uses **Qwen2.5-0.5B-Instruct** and short prompts (`sharegpt_edge_v1`).
It is aimed at single-GPU consumer hardware (≥4 GB VRAM). On pre-Ampere GPUs,
use `--enforce-eager` with the NVIDIA vLLM runner — see
[runners/nvidia_vllm_47f5d58e/README.md](runners/nvidia_vllm_47f5d58e/README.md).

```bash
python run.py --runner nvidia_vllm_47f5d58e --suite suite_F
```

Full specs: [suites/README.md](suites/README.md#suite-f).

**Note on `concurrency` vs batch size:** The offline scenario sweeps
*client-side concurrency* (how many requests the load generator fires
simultaneously) — not the inference engine's internal batch size.
The engine's internal batching (e.g. vLLM's `max_num_seqs`) is
configured separately in the runner and is not varied by the suite.
Results report `concurrency` values, not batch sizes.

### With a local model path

```bash
python run.py --runner nvidia_vllm_47f5d58e \
    --suite suite_A \
    --model-path /path/to/local/model
```

---

## What gets measured

### Scenarios

| Scenario | Primary metric | What it tells you |
|----------|---------------|-------------------|
| **offline** | tokens/sec | Max throughput when the GPU is fully loaded |
| **online** | max valid QPS | How many users/sec this chip can serve within a latency SLA |
| **interactive** | TTFT p99 | Single-user latency when the system is idle |
| **sustained** | sustained tok/s + throttle ratio | Whether throughput degrades over a long fixed-concurrency run (typically 30 min; Suite F uses 15 min) |
| **accuracy** | subset score | Model output quality against an MMLU baseline |
| **speculative** *(extra)* | tokens/sec + acceptance rate | Offline throughput with speculative decoding (draft model); also captures `task.runtime_metrics.acceptance_rate` if the runner implements `get_runtime_metrics()`. Suites A and D. |
| **burst** *(extra)* | burst degradation ratio | TTFT p99 during burst windows vs steady windows; stress-tests KV cache eviction and scheduler resilience. Suites A and B. |

Which scenarios are included in the default run and which are extras depends on
the suite. Check the suite's `suite.json` or see [suites/README.md](suites/README.md).

> **Running speculative on Suite A or D:** the draft model path is resolved
> automatically from `speculative_draft_model_id` in the suite config (respecting
> `configs/models_local.yaml`). No manual engine config is required.

### Key metrics

**`throughput_tokens_per_sec`** — input + output tokens per second. Offline
scenario primary metric.

**`max_valid_qps`** — highest request rate (queries/sec) where TTFT p99 stays
under the suite's SLA. Online scenario primary metric.

**`ttft_ms_p99`** — 99th percentile time-to-first-token in milliseconds.
Interactive scenario primary metric.

**`throttle_ratio`** — `min_throughput / max_throughput` over the 30-minute
sustained run. 1.0 = perfectly stable. A100 reference: ~0.87 at concurrency 8.

**`quality_efficiency`** — Suite C only. `speedup_vs_bf16 × accuracy_retention`.
Rewards quantized formats that are both fast and accurate.
See [suites/README.md](suites/README.md) for full details on all suites and metrics.

### Expected run times (A100-SXM4-80GB reference)

| Suite | Default run | Notes |
|-------|-------------|-------|
| A | ~11 min | offline + online |
| B | ~20 min | 4× chips (A100-80GB); ~37 min on A100-40GB |
| C | ~22 min | offline only, all 5 precision formats |
| D | ~22 min | Long-context (offline only) |
| E | ~9 min | Up to 4× scaling |
| F | ~10 min | offline + online + interactive |
| G | ~35 min | MoE multi-chip (varies with chip count) |

Sustained adds about **30 minutes** on datacenter suites (A–E); **Suite F** uses a **15-minute** sustained profile.
Add interactive (~35 min on Suite A) by running with `--scenario interactive` or `--scenario all`.
Add **speculative** (~3 min extra on Suite A, ~24 min extra on Suite D) or **burst** (~18 min extra on Suite A/B) with `--scenario speculative` / `--scenario burst`.

---

## Submitting your results

### Accuracy gate (automatic)

When you run the suite, accuracy runs automatically as the **first step**.
If accuracy fails, the benchmark is aborted.

```
============================================================
  Step 1: Accuracy Gate
  Must pass before benchmark runs.
============================================================

Score: 62/100 = 0.6200
Baseline: 0.6000
Delta: +0.0200 (min allowed: -0.10 — matches `accuracy_threshold_delta` in many suites; Suite C uses per-format thresholds)
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

### Step 1: Open a GitHub Issue

Go to [Issues → New → Community Submission](https://github.com/JuhaoLiang1997/AccelMark/issues/new?template=community_submission.md).

Paste the full contents of your `result.json` into the code block and submit.

> **The CI bot validates your result automatically** — recommend to run
> `validate_submission.py` locally first. If validation fails, the bot
> comments on your issue explaining what to fix.

> **Why paste instead of attach?** The CI bot reads `result.json` directly
> from the issue body. File attachments are not accessible to GitHub Actions.

### Step 2: Done

The CI bot will:
1. Validate your `result.json` against the schema
2. Open a PR with your result files
3. Comment on your issue with a link to the PR

Your result appears on the **Community** tab after the maintainer reviews
and merges the PR — usually within a day or two.

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
python run.py --runner nvidia_vllm_47f5d58e --model-path /your/path/...
```

If your local copy was downloaded at a different revision, add a note in
`meta.notes` of `result.json`.

---

## Adding support for a new platform

Create a new runner folder under `runners/` by subclassing `BenchmarkRunner`.
See [DEVELOPMENT.md](DEVELOPMENT.md) for the full implementation guide including
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

All orchestration (result building, accuracy reuse, Suite E, etc.) is
inherited from `BenchmarkRunner` automatically.

**Checklist for a new platform PR:**
- [ ] Runner folder named `{platform}_{name}_{hash8}` with correct hash
- [ ] `runner.py` subclasses `BenchmarkRunner` and passes `runners/validate_runners.py`
- [ ] `meta.json` present and valid (see `runners/meta.schema.json`)
- [ ] `requirements.txt` included
- [ ] At least one reference result in `results/community/`
- [ ] `runners/collect_env.py` updated to detect your hardware (see [DEVELOPMENT.md](DEVELOPMENT.md))
- [ ] `README.md` supported platforms table updated

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full implementation reference.

---

## Reporting a suspicious result

If a result looks wrong:

1. Open a GitHub Issue using the **"Challenge a Result"** template
2. Include: the submission name, what looks wrong, and ideally your own
   run on the same hardware as evidence

Maintainers will investigate. If confirmed suspicious, the result's `meta.flagged`
field will be set to a reason string and it will appear with a ⚠️ badge on the
leaderboard.

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
  underrepresent true hardware capability. `--enforce-eager` is a runner-specific
  flag (not a base `run.py` flag); it can also be set permanently via
  `enforce_eager: true` in the runner config yaml at
  `configs/runner_configs/runner_<id>.yaml`.
- Results submitted with `--skip-accuracy-gate` are permanently flagged