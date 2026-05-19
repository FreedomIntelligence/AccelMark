# Contributing to AccelMark

AccelMark is a community-driven benchmark leaderboard for AI accelerators.
The most valuable contribution is running the benchmark on hardware not yet
in the leaderboard and submitting your results.

---

## Quick start

**Got a GPU? Here's the shortest path to getting on the leaderboard:**

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<you>/AccelMark.git
cd AccelMark
pip install -e .
pip install -r runners/nvidia_vllm_47f5d58e/requirements.txt

# 2. Set your name (one-time setup)
cp configs/submitter.yaml.example configs/submitter.yaml
# Edit configs/submitter.yaml — add your name or GitHub username

# 3. Run the benchmark (~11 min on A100 for default scenarios)
#    Accuracy gate runs automatically before the benchmark starts.
#    Output directory is auto-named using run_name, e.g.:
#    results/community/nvidia_a100_sxm4_80gbx1_suite_A_nvidia_vllm_47f5d58e_ed4b0557
python run.py --runner nvidia_vllm_47f5d58e --suite suite_A

# 4. Open a pull request with your result
git checkout -b submit/<your-hardware>
git add results/ && git commit -m "results: <hardware> on suite_A"
git push origin submit/<your-hardware>
gh pr create   # or open the PR via the GitHub web UI
```

That's it. CI validates the result automatically; merging the PR publishes it to the leaderboard.

> _Prefer not to use git?_ Open a [Community Submission issue](https://github.com/FreedomIntelligence/AccelMark/issues/new?template=community_submission.md), paste your `result.json`, and the CI bot will draft the PR on your behalf.

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

### Per-runner config overrides (optional)

If you want to permanently change a runner's defaults (e.g. raise
`max_num_seqs`, enable `enforce_eager`, set `tensor_parallel_size`) without
adding flags to every invocation, drop a yaml at
`configs/runner_configs/runner_<runner_id>.yaml`. The file is
**gitignored** — only `*.yaml.example` companions are checked into the
repo. That makes the override strictly local to your machine and keeps
the canonical defaults intact for everyone else.

```bash
cp configs/runner_configs/runner_nvidia_vllm_47f5d58e.yaml.example \
   configs/runner_configs/runner_nvidia_vllm_47f5d58e.yaml
# edit freely — your benchmarks now pick up the overrides automatically
```

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

## Submitting a result

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

### Recommended: open a pull request

After a successful run, validate locally and open a PR:

```bash
# Validate the produced files against the schemas (the same check CI runs).
python runners/validate_submission.py \
    results/community/<run_name>/result.json

# Stage just the new result and env file.
git checkout -b submit/<your-hardware>
git add results/community/<run_name>/
git commit -m "results: <hardware> on suite_A"
git push origin submit/<your-hardware>

# Open the PR — either via the GitHub web UI or:
gh pr create --fill
```

What gets committed is *only* the new files under `results/community/<run_name>/`:
your `result.json`, `env_info.json`, and (optionally) `samples.jsonl`. Nothing
else in the repo should change.

CI then re-runs the schema validator and the runner-folder integrity check.
When both pass and a contributor reviews the diff, the PR is merged and your
result shows up on the leaderboard on the next site build.

### Optional: preview the leaderboard locally

The static site is generated from `results/` by `leaderboard/generate.py`.
After dropping your result into `results/community/<run_name>/`, you can
preview the final UI before opening the PR:

```bash
python leaderboard/generate.py                       # writes leaderboard/site/leaderboard.js + api/
python -m http.server -d leaderboard/site 8000       # serve the static site
# open http://localhost:8000
```

Both `leaderboard.js` and `leaderboard/site/api/` are gitignored — the GitHub
Actions workflow regenerates them on every merge to `main`.

### Alternative: open a submission issue (no git required)

If you'd rather not use git, paste your `result.json` into a
[Community Submission issue](https://github.com/FreedomIntelligence/AccelMark/issues/new?template=community_submission.md).
A bot will validate the JSON, draft a PR with the files in the right place,
and link it back to your issue. You don't need to touch git or fork the repo.

> **Why paste instead of attach?** The bot reads `result.json` directly from
> the issue body. File attachments are not accessible to GitHub Actions.

---

## Leaderboard tiers

| Tier | How to get it | Leaderboard placement |
|------|--------------|----------------------|
| **community** | Submit a PR (or issue → bot-drafted PR) and pass CI validation | Community tab |
| **verified** | Independently reproduced on the same hardware/runner within 5% | Main leaderboard |

To promote a community result to **verified**, anyone with the same hardware
and runner can run the same suite and open a follow-up PR that lands the
reproduction in `results/verified/`. Maintainers do not gate this — every
verified result is itself reproducible by definition.

---

## How your result appears on the leaderboard

The frontend treats every submission as data — there's nothing per-vendor or per-chip hand-coded on the UI side. A few conventions are worth knowing if you want your result to look its best:

### Chip identity vs. chip count

The chip-detail page (`#/chip/<slug>`) is keyed on the **chip model** alone, not on `chip_count`. That means a single chip page aggregates every fan-out you've ever submitted (×1, ×4, ×8, ×16) into one overview, with the runs table sorted by `chip_count` ascending and the per-suite KPI card flagging the deployment behind the best score (e.g. `×8` badge next to the metric).

Implication: if your runner emits one `result.json` per chip-count, you don't need to invent fake chip names to keep them apart — submit them with the same `chip` field and they'll merge cleanly. Old `…-x<N>` URLs from before this change auto-redirect to the bare-model slug, so existing shared links keep working.

### Vendor colours

Vendor accents (chip dot, vendor pill, peer card border) are driven entirely by `assets/js/data.js`'s `VENDOR_COLORS` map. New vendors get a deterministic fallback colour from a 9-entry palette the first time they appear in the dataset — no frontend change required to ship the result.

If you want a brand-accurate accent for a new vendor (e.g. your accelerator's official colour), open a one-line PR to `VENDOR_COLORS`:

```js
export const VENDOR_COLORS = {
  // …
  Cerebras: "#ff6f3c",   // ← add yours
};
```

`VENDOR_ORDER` (used to lay out the rankings facet pill row) is derived from `Object.keys(VENDOR_COLORS)`, so the same edit also pins your vendor's position in the brand list. Vendors not in the map are appended alphabetically after the brand-named ones.

### Optional: viz fields for richer modal charts

The run-detail modal's **Visualize** tab is hidden when `result.viz` is absent. Populate it to surface scenario-specific charts:

```jsonc
{
  "viz": {
    "type": "decode",            // bandwidth-bound suite (A/F/G default)
    "offline": {
      "labels":     [1, 2, 4, 8, 16, 32],
      "throughput": [120, 230, 410, 760, 1100, 1380]
    },
    "interactive": {             // optional, suite_D-style
      "ttft_p50": 78,  "ttft_p90": 110, "ttft_p99": 135,
      "tpot_p50":  9,  "tpot_p90":  11, "tpot_p99":  14
    }
  }
}
```

Suite-specific shapes the frontend understands today:

| `viz.type` | Suites | Required keys |
|---|---|---|
| `decode`    | A · F · G  | `offline.labels[]`, `offline.throughput[]` |
| `multichip` | B          | `offline.labels[]`, `offline.throughput[]`, `offline.throughput_per_chip[]` |
| `quant`     | C          | `precisions[]`, `throughput[]`, optional `accuracy[]` |
| `longctx`   | D          | `offline.labels[]`, `offline.throughput[]`, `interactive.{ttft_p50,…,tpot_p99}` |
| `scaling`   | E          | `chip_counts[]`, `throughput[]`, `efficiency_pct[]` |

`viz` is **fully optional** — runners that don't emit it still get a clean Details / Implementation modal. When present, the same fields drive the per-suite head-to-head charts on the Compare page (so two basket runs render directly comparable visualisations instead of falling back to a metric-table-only view).

### Submitter handle

The leaderboard surfaces the value of `meta.submitted_by` as `@<handle>` next to your result on every list (home recent, suite cards, chip-detail submissions table). Anything that looks like a GitHub login, an email, or a `Name <email>` form is reduced to the local-part — see `submitterHandle` in `assets/js/utils.js`.

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

## Adding a new runner

A "runner" here is a Python class that wraps an inference framework (vLLM,
SGLang, mlx-lm, …) and exposes the AccelMark standard interface. Adding
one for an **existing** platform (NVIDIA, AMD, Ascend, Apple, Google TPU,
Moore Threads, …) does not require touching any shared file. The full
walk-through lives in [`runners/README.md`](runners/README.md); the short
version is:

1. Copy `runners/template/runner.py` into a temporary folder and fill in
   the three required methods (`load_model`, `inference_fn_offline`,
   `release_resources`) plus `inference_fn_streaming` if your framework
   has a streaming API.
2. Compute the hash and rename the folder:
   `python runners/hash_runner.py runners/tmp/`
   produces e.g. `nvidia_myframework_3f8a2c1d`.
3. Write `meta.json` next to it, including `suite_support` — that field
   is **how the top-level `README.md` table picks up your runner**. You
   never edit `README.md` yourself.
4. Add a `requirements.txt`.
5. Validate: `python runners/validate_runners.py --dir runners/<your_folder>`.
6. Regenerate the README matrix locally:
   `python tools/generate_platforms_matrix.py`.

```python
# runners/your_platform_{hash8}/runner.py
from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
from loadgen.types import InferenceResult

class MyFrameworkRunner(BenchmarkRunner):

    SUPPORTS_STREAMING  = True    # set False if no streaming API
    SUPPORTS_BATCHING   = True    # set False if serial only (e.g. mlx-lm)
    SUPPORTS_ONLINE     = True
    SUPPORTS_MULTI_CHIP = True    # set False if no tensor parallelism

    def load_model(self, model_path: str, parallelism: dict) -> None:
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

**Checklist for a new-runner PR (existing platform):**
- [ ] Runner folder named `{platform}_{name}_{hash8}` with correct hash
- [ ] `runner.py` subclasses `BenchmarkRunner` and passes `runners/validate_runners.py`
- [ ] `meta.json` present and valid (see `runners/meta.schema.json`), with
      `suite_support` declared for every suite your runner can or cannot run
- [ ] `requirements.txt` included
- [ ] `tools/generate_platforms_matrix.py --check` passes locally (CI also
      enforces this)
- [ ] At least one reference result in `results/community/` (validated by CI)

### Adding a new accelerator family

If you are bringing up a **new platform** (e.g. a vendor not yet in
`schema/platforms.json`), the only additional file you need to ship is

```
runners/platforms/<your_platform>.py
```

which exports module-level `collect()`, `detect_runtime_version()` and a
few optional helpers. The collector at `runners/collect_env.py`
auto-discovers it; no change to that file is required. See
[`runners/README.md`](runners/README.md#adding-a-new-accelerator-family)
for the full protocol and a worked example.

Optional polish steps when the new platform stabilises:

- Add an entry to `schema/platforms.json` so the README matrix renders a
  pretty hardware label and stable sort order. Until then, the matrix
  renders the bare identifier and `validate_runners.py` emits a
  non-fatal warning prompting this follow-up.

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full implementation reference.

---

## Reporting a suspicious result

If a result looks wrong:

1. Open a GitHub Issue using the **"Challenge a Result"** template
2. Include: the submission name, what looks wrong, and ideally your own
   run on the same hardware as evidence

The community discusses the report in the issue. If the consensus is that
the result is suspicious, a PR sets `meta.flagged` on that result to a
reason string and the entry shows up with a ⚠️ badge on the leaderboard.
Anyone can open that follow-up PR.

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