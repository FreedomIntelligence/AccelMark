# AccelMark — Developer Guide

This document is for contributors who want to extend AccelMark:
adding a new inference framework, a new suite, a new chip platform,
or modifying the leaderboard pipeline.

For running benchmarks and submitting results, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Architecture Overview

```
AccelMark/
├── run.py                  ← Unified CLI entry point
├── runners/
│   ├── benchmark_runner.py ← Base class: all orchestration logic
│   ├── collect_env.py      ← Hardware/software detection
│   ├── validate_submission.py
│   ├── hash_runner.py      ← Compute runner ID before submission
│   ├── validate_runners.py ← CI: validate all runner folders
│   ├── meta.schema.json    ← JSON schema for runner meta.json
│   └── nvidia_vllm_{hash}/ ← Reference runner (NVIDIA + vLLM)
│       ├── runner.py
│       ├── requirements.txt
│       └── meta.json
├── loadgen/
│   ├── loadgen.py          ← Shared timing and measurement engine
│   └── types.py            ← InferenceResult, SampleRecord
├── suites/
│   ├── suite_A/suite.json + requests.jsonl
│   ├── suite_B/...
│   ├── suite_C/...
│   ├── suite_D/...
│   └── suite_E/...
├── schema/
│   ├── result.schema.json
│   ├── accuracy_subset.jsonl   ← immutable
│   └── accuracy_baselines.json
├── leaderboard/
│   ├── generate.py         ← reads results/, writes leaderboard.js + api/
│   └── site/
│       └── index.html
└── results/
    ├── verified/
    └── community/
```

### Data flow

```
run.py  (or direct: python runners/{id}/runner.py)
    ↓  loads runner by ID
runners/{id}/runner.py  (BenchmarkRunner subclass)
    ↓  calls
BenchmarkRunner._run_single_scenario()
    ↓  calls
AccelMarkLoadGen.run(inference_fn)          ← loadgen handles all timing
    ↓  returns metrics dict
BenchmarkRunner._build_result_json()       ← assembles result.json
    ↓  writes
results/community/{name}/result.json

GitHub Actions
    ↓  on push to results/
leaderboard/generate.py                    ← reads all result.json files
    ↓  writes
leaderboard/site/leaderboard.js
leaderboard/site/api/index.json            ← queried by OpenClaw Skill
```

### Key design principle

**LoadGen owns all timing.** Platform scripts never measure time themselves.
`loadgen.py` controls when requests are sent, when results are collected,
and what metrics are computed. This ensures all results are comparable
regardless of platform.

---

## Adding a New Inference Framework

### Overview

Create a new runner folder under `runners/{platform}_{framework}_{hash8}/` that
subclasses `BenchmarkRunner` and implements three methods.

### Step 0: Compute the runner ID

Before naming your folder, write your `runner.py` first, then compute the hash:

    python runners/hash_runner.py path/to/your/runner.py

This prints your implementation ID, e.g. `nvidia_lmdeploy_7f3a1b2c`.
Create your folder with that exact name.

### Step 1: Implement the subclass

```python
# runners/nvidia_lmdeploy_{hash8}/runner.py
from runners.benchmark_runner import BenchmarkRunner
from loadgen.types import InferenceResult
import time

class LMDeployRunner(BenchmarkRunner):

    # ── Declare capabilities ──────────────────────────────────────────
    SUPPORTS_STREAMING = True    # LMDeploy supports streaming
    SUPPORTS_BATCHING = True     # LMDeploy supports batch inference
    SUPPORTS_ONLINE = True
    SUPPORTS_MULTI_CHIP = True

    def __init__(self):
        self.pipeline = None
        self.tokenizer = None

    # ── Required: model loading ───────────────────────────────────────
    def load_model(self, model_path: str, suite: dict, tp_size: int) -> None:
        from lmdeploy import pipeline, TurbomindEngineConfig
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = pipeline(
            model_path,
            backend_config=TurbomindEngineConfig(tp=tp_size),
        )

    # ── Required: offline batch inference ────────────────────────────
    def inference_fn_offline(self, prompts: list[str]) -> list[InferenceResult]:
        formatted = [self._format_prompt(p) for p in prompts]
        t_start = time.perf_counter()
        outputs = self.pipeline(formatted)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return [
            InferenceResult(
                first_token_time_ms=None,
                total_time_ms=elapsed_ms,
                output_tokens=len(o.token_ids),
                input_tokens=len(o.input_token_ids),
                success=True,
            )
            for o in outputs
        ]

    # ── Required: resource cleanup ────────────────────────────────────
    def release_resources(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

    # ── Optional: streaming for online/interactive ───────────────────
    async def inference_fn_streaming(self, prompt: str) -> InferenceResult:
        import asyncio
        formatted = self._format_prompt(prompt)
        t_start = time.perf_counter()
        first_token_time_ms = None
        output_tokens = 0

        async for output in self.pipeline.stream_infer(formatted):
            if first_token_time_ms is None:
                first_token_time_ms = (time.perf_counter() - t_start) * 1000
            output_tokens = output.num_tokens

        return InferenceResult(
            first_token_time_ms=first_token_time_ms,
            total_time_ms=(time.perf_counter() - t_start) * 1000,
            output_tokens=output_tokens,
            input_tokens=0,
            success=True,
        )

    # ── Optional: memory query ────────────────────────────────────────
    def get_peak_memory_gb(self) -> float:
        import torch
        try:
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            return None

    # ── Optional: framework metadata ─────────────────────────────────
    def _get_framework_name(self) -> str:
        return "LMDeploy"

    def _get_framework_version(self) -> str:
        try:
            import lmdeploy
            return lmdeploy.__version__
        except Exception:
            return "unknown"

    def _format_prompt(self, prompt: str) -> str:
        if self.tokenizer and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt


if __name__ == "__main__":
    LMDeployRunner().main()
```

### Step 2: Add requirements

```
# runners/nvidia_lmdeploy_{hash8}/requirements.txt
lmdeploy>=0.5.0
transformers>=4.40.0
```

### Step 3: Add a README

```
# runners/nvidia_lmdeploy_{hash8}/README.md

## Setup

pip install lmdeploy>=0.5.0

## Usage

python run.py --runner nvidia_lmdeploy_{hash8} --suite suite_A --scenario all
```

### Step 4: Test it

```bash
# Verify imports
python -c "from runners.nvidia_lmdeploy_{hash8}.runner import LMDeployRunner; print('OK')"

# Dry run
python run.py --runner nvidia_lmdeploy_{hash8} --help

# Full run
python run.py --runner nvidia_lmdeploy_{hash8} --suite suite_A --scenario all
```

### Step 5: Write meta.json

    {
      "id":           "nvidia_lmdeploy_7f3a1b2c",
      "platform":     "nvidia",
      "name":         "LMDeploy on NVIDIA",
      "framework":    "LMDeploy",
      "submitted_by": "your_github_username",
      "description":  "One sentence describing what makes this runner distinct.",
      "notes":        null,
      "created":      "YYYY-MM-DD"
    }

The `id` must exactly match the folder name.

### Capability flags

Override these class attributes to declare what your framework supports:

| Flag | Default | When to set False |
|------|---------|------------------|
| `SUPPORTS_STREAMING` | `True` | Framework has no token streaming API. TTFT will not be measured for online/interactive. |
| `SUPPORTS_BATCHING` | `True` | Framework is serial only (e.g. mlx-lm). Offline runs requests one-by-one. |
| `SUPPORTS_ONLINE` | `True` | Framework cannot handle concurrent requests. Online scenario is skipped. |
| `SUPPORTS_MULTI_CHIP` | `True` | No tensor parallelism support. `--tensor-parallel-size` is ignored. |

### Example: Apple Silicon (no batching, no streaming)

```python
class MLXRunner(BenchmarkRunner):
    SUPPORTS_STREAMING = False   # mlx-lm has no streaming API
    SUPPORTS_BATCHING = False    # serial only
    SUPPORTS_MULTI_CHIP = False  # no tensor parallelism

    def load_model(self, model_path, suite, tp_size):
        from mlx_lm import load
        self.model, self.tokenizer = load(model_path)

    def inference_fn_offline(self, prompts):
        # SUPPORTS_BATCHING=False: loadgen calls this one prompt at a time
        from mlx_lm import generate
        assert len(prompts) == 1
        prompt = prompts[0]
        t_start = time.perf_counter()
        output = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=512)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        tokens = len(self.tokenizer.encode(output))
        return [InferenceResult(
            first_token_time_ms=None,
            total_time_ms=elapsed_ms,
            output_tokens=tokens,
            input_tokens=0,
            success=True,
        )]

    def release_resources(self):
        del self.model
        self.model = None
```

---

## Adding a New Suite

Suites are fully specified benchmark configurations. Each suite answers
one question — variables are controlled, one dimension changes at a time.

### Step 1: Design the suite

Before writing any files, answer these questions:

```
1. What question does this suite answer?
   e.g. "How does this chip handle quantized 8B inference?"

2. What is the controlled variable?
   e.g. quantization level (INT8 vs INT4 vs BF16)

3. What model?
   Use a model that is already in another suite if possible.
   New models require downloading and generating new requests.jsonl.

4. What scenarios?
   offline: always include (throughput is the most comparable metric)
   online: include if latency under load matters
   interactive: include if single-user latency matters

5. What chip count?
   1 chip: for suites that test per-chip capability
   flexible: for suites where chip count is part of the experiment
```

### Step 2: Create `suites/suite_X/suite.json`

Copy the closest existing suite and modify. Required fields:

```json
{
  "suite_id": "suite_X",
  "description": "One sentence describing what this suite measures.",
  "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
  "model_revision": "8afb486c...",
  "scenarios": ["offline", "online", "interactive"],
  "precision_required": "BF16",
  "request_distribution": {
    "input_tokens_p50": 280,
    "output_tokens_p50": 310,
    "source": "shibing624/sharegpt_gpt4"
  },
  "output_tokens_max": 512,
  "concurrency_levels": [8, 32, 128],
  "online_qps_levels": [5, 10, 25, 50, 100],
  "online_sla_ttft_ms": 500,
  "num_runs": 3,
  "warmup_runs": 1,
  "online_warmup_runs": 0,
  "interactive_warmup_runs": 0,
  "accuracy_threshold_delta": 0.03,
  "request_count": 200,
  "online_request_count": 500,
  "interactive_request_count": 100
}
```

### Step 3: Generate `suites/suite_X/requests.jsonl`

```bash
# Option A: same distribution as Suite A — just copy it
cp suites/suite_A/requests.jsonl suites/suite_X/requests.jsonl

# Option B: generate from ShareGPT with custom parameters
# Edit AccelMark-internal/data_pipeline/configs/suite_X.yaml
# then run:
python AccelMark-internal/data_pipeline/generate_requests.py --suite suite_X
bash AccelMark-internal/ops/publish_outputs.sh suite_X
```

Format of each line in `requests.jsonl`:
```json
{
  "request_id": 0,
  "prompt": "...",
  "input_tokens": 245,
  "conversation_id": "sg_00001",
  "turn_index": 0,
  "prompt_type": "conversational"
}
```

### Step 4: Add accuracy baseline

Run the accuracy check on reference hardware (A100) and record the score:

```bash
python run.py --runner nvidia_vllm_e0859b3c \
    --suite suite_X \
    --scenario accuracy \
    --model-path /path/to/model
```

Add to `schema/accuracy_baselines.json`:
```json
{
  "meta-llama/Meta-Llama-3-8B-Instruct": {
    "revision": "8afb486c...",
    "bf16_baseline_score": 0.62
  }
}
```

### Step 5: Document in `suites/README.md`

Add a section following the same format as existing suites:

```markdown
## Suite X

**One-line description**

> *"The question this suite answers?"*

| | |
|---|---|
| **Model** | ... |
| **Chips** | ... |
| **Scenarios** | ... |
| **Primary metrics** | ... |
| **Run time** | ... |
```

### Step 6: Submit a reference result

Before announcing the suite, submit at least one verified result
from reference hardware. New suites without reference results are
not shown on the main leaderboard.

---

## Adding a New Platform (Chip Type)

Adding a new platform means adding support for a chip family that
`collect_env.py` doesn't recognize yet.

### Step 1: Add hardware detection to `collect_env.py`

`collect_env.py` has four existing collectors:
- `collect_nvidia()` — uses `nvidia-smi`
- `collect_amd()` — uses `rocm-smi`
- `collect_ascend()` — uses `npu-smi`
- `collect_apple()` — uses `system_profiler`

Add a new function for your platform:

```python
def collect_your_platform() -> list[dict]:
    """Detect YourPlatform accelerators."""
    accelerators = []
    try:
        # Use your platform's CLI tool to query hardware
        output = subprocess.check_output(
            ["your-smi", "--query", "--format=json"],
            text=True
        )
        data = json.loads(output)
        for device in data["devices"]:
            accelerators.append({
                "name": device["name"],
                "vendor": "YourVendor",
                "memory_gb": device["memory_mb"] / 1024,
                "driver_version": device["driver_version"],
                "runtime_version": device.get("sdk_version"),
                "compute_capability": None,
                "pcie_generation": None,
                "interconnect_intra_node": device.get("interconnect"),
            })
    except Exception as e:
        print(f"Warning: could not detect YourPlatform: {e}")
    return accelerators
```

Add detection to the `main()` dispatcher:

```python
def main():
    ...
    # Detection order: nvidia → amd → ascend → apple → your_platform
    accelerators = (
        collect_nvidia() or
        collect_amd() or
        collect_ascend() or
        collect_apple() or
        collect_your_platform() or
        []
    )
```

### Step 2: Create the platform script

```
runners/your_platform_{hash8}/
├── runner.py        ← BenchmarkRunner subclass
├── requirements.txt
├── meta.json
└── README.md
```

### Step 3: Update the supported platforms table

In `README.md`, add your platform to the supported platforms table:

```markdown
| YourVendor (ModelX) | YourFramework | ✓ | — | — | — |
```

### Step 4: Update `schema/cloud_pricing.json`

If your chip is available on cloud providers, add pricing:

```json
"YourVendor ModelX 80GB": {
    "providers": [
        {
            "name": "CloudProvider (per GPU)",
            "price_usd_per_hr": 3.50,
            "source": "cloudprovider.com/pricing",
            "updated": "2026-03"
        }
    ]
}
```

---

## The LoadGen Contract

`loadgen/loadgen.py` is the core timing engine. **Do not modify it for
platform-specific reasons.** All platforms use the same LoadGen.

### What LoadGen expects from inference_fn

**Offline scenario** (sync):
```python
def inference_fn(prompts: list[str]) -> list[InferenceResult]:
    # Must return one InferenceResult per prompt
    # Must complete all prompts before returning
    # Do NOT time anything inside this function
    # LoadGen handles all timing
```

**Online and interactive scenarios** (async):
```python
async def inference_fn(prompt: str) -> InferenceResult:
    # Must be a coroutine (async def)
    # LoadGen schedules concurrent calls for online
    # LoadGen awaits serially for interactive
    # first_token_time_ms should be set if streaming is available
```

### InferenceResult fields

```python
@dataclass
class InferenceResult:
    first_token_time_ms: Optional[float]  # None if streaming not supported
    total_time_ms: float                  # wall clock from request to completion
    output_tokens: int                    # number of generated tokens
    input_tokens: int                     # number of input tokens (0 if unknown)
    success: bool                         # False if inference failed
    error: Optional[str] = None           # error message if success=False
```

### What LoadGen measures

| Scenario | Measures | Primary metric |
|----------|----------|----------------|
| offline | Total tokens / elapsed time | `throughput_tokens_per_sec` (input + output) |
| online | TTFT distribution at each QPS level | `max_valid_qps` (highest QPS with p99 TTFT < SLA) |
| interactive | TTFT distribution, serial requests | `ttft_ms_p99` |

---

## Schema and Validation

### result.schema.json

All result.json files are validated against `schema/result.schema.json`
before being accepted into the leaderboard.

Key constraints:
- `task` must have either `scenario` (single run) or `scenarios_run` (suite-level)
- `accuracy.valid` must be `true` for verified tier
- `submitted_by` must be non-empty
- `metrics` fields that are null are allowed (power, memory)

When adding new fields to result.json, update the schema to allow them.
Use `"type": ["your_type", "null"]` to make fields optional.

### Adding a new field to result.json

1. Add to `_build_result_json()` in `benchmark_runner.py`
2. Add to `schema/result.schema.json` as optional (`"type": ["X", "null"]`)
3. Add to `extract_row()` in `leaderboard/generate.py` if it should appear on leaderboard
4. Run `validate_submission.py` on an existing result to confirm backward compatibility

### Runner validation (`validate_runners.py`)

Before opening a PR that adds a new runner, validate it locally:

```bash
python runners/validate_runners.py runners/nvidia_vllm_3f8a2c1d/
```

This validates a single runner folder and tells you clearly whether it is
ready to submit:

```
Validating: nvidia_vllm_3f8a2c1d/
==================================================
Files:
  ✓ runner.py
  ✓ meta.json
  ✓ requirements.txt

Hash:
  ✓ SHA-256(runner.py)[:8] = 3f8a2c1d ✓

meta.json:
  ✓ Valid against schema
  ✓ meta.id matches folder name

Duplicate check:
  ✓ No existing runner with this ID

==================================================
✓ PASSED — nvidia_vllm_3f8a2c1d is ready to submit
==================================================
```

| Check | What it enforces |
|-------|-----------------|
| `runner.py` present | Every runner folder must have a runnable entry point |
| `meta.json` present | Metadata is required for discovery and the leaderboard |
| Hash consistency | Folder name must end with `SHA-256(runner.py)[:8]` — detects untracked edits |
| `meta.json` schema | Validates required fields: `id`, `platform`, `name`, `framework`, `submitted_by`, `description` |
| `meta.id == folder name` | The ID in metadata must exactly match the folder name |
| No duplicate IDs | Checks that no existing runner in `runners/` shares the same ID |
| `supersedes` target exists | Warning if the referenced old runner folder is not found |
| `deprecated_by` target exists | Warning if the referenced new runner folder is not found |

`requirements.txt` absence is a warning, not an error.
`supersedes` and `deprecated_by` cross-reference failures are also warnings —
the referenced folder may not be merged yet when validating locally.

**Hash mismatch** is the most common failure after editing `runner.py` without
renaming the folder. The error message tells you exactly what to do:

```
  ✗ Hash mismatch.
      Folder ends with : e0859b3c
      runner.py hashes to: b3f29a11
      Rename folder to: nvidia_vllm_b3f29a11
```

To compute the correct name before creating a new runner folder:

```bash
python runners/hash_runner.py path/to/your/runner.py
# → nvidia_vllm_b3f29a11
```

CI runs the same validator across all runner folders automatically on every PR.

### Updating an existing runner

Runner folders are immutable once merged — you cannot edit `runner.py` in
place. Instead, publish a new folder and mark the old one deprecated.
This preserves the audit trail: results that reference the old ID always
point to the exact code that produced them.

**Step 1: Edit your `runner.py` and compute the new hash**

```bash
# Make your changes, then compute the new ID
python runners/hash_runner.py runners/nvidia_vllm_old_hash/runner.py
# → nvidia_vllm_b3f29a11
```

**Step 2: Create the new runner folder**

```bash
cp -r runners/nvidia_vllm_old_hash runners/nvidia_vllm_b3f29a11
# Apply your edits to runners/nvidia_vllm_b3f29a11/runner.py
```

**Step 3: Update `meta.json` in the new folder**

```json
{
  "id":           "nvidia_vllm_b3f29a11",
  "platform":     "nvidia",
  "name":         "vLLM on NVIDIA (reference implementation)",
  "framework":    "vLLM",
  "submitted_by": "JuhaoLiang1997",
  "description":  "...",
  "supersedes":   "nvidia_vllm_old_hash",
  "notes":        null,
  "created":      "YYYY-MM-DD"
}
```

**Step 4: Add `deprecated_by` to the old runner's `meta.json`**

`meta.json` is the only file that may be edited in an existing runner folder.

```json
{
  "id":            "nvidia_vllm_old_hash",
  "deprecated_by": "nvidia_vllm_b3f29a11",
  "notes":         "Deprecated — use nvidia_vllm_b3f29a11. Fixed edge case in release_resources()."
}
```

**Step 5: Validate and submit**

```bash
python runners/validate_runners.py runners/nvidia_vllm_b3f29a11/
```

Open a PR that includes both the new folder and the updated old `meta.json`.
The old runner remains runnable — existing results are unaffected. `run.py --list`
will hide it by default and show a deprecation warning if someone runs it directly.

---

## Testing Your Changes

### Before submitting a PR

```bash
cd /path/to/AccelMark

# 0. Validate your runner folder (hash, meta.json schema, no duplicate IDs)
python runners/validate_runners.py runners/your_platform_{hash8}/

# 1. Schema is valid JSON
python -c "import json; json.load(open('schema/result.schema.json')); print('schema OK')"

# 2. Existing results still validate
for dir in results/verified/*/; do
    python runners/validate_submission.py --dir "$dir" && echo "OK: $dir"
done

# 3. Leaderboard generates without errors
python leaderboard/generate.py

# 4. New runner imports cleanly
python -c "from runners.your_platform_{hash8}.runner import YourRunner; print('OK')"

# 5. Help works
python run.py --runner your_platform_{hash8} --help
```

### Running a quick benchmark test

```bash
# Run with minimal requests to test the pipeline end-to-end
# Temporarily reduce request_count for testing only
python run.py --runner nvidia_vllm_e0859b3c \
    --suite suite_A \
    --scenario offline \
    --output-dir /tmp/accelmark_test/

# Validate the output
python runners/validate_submission.py --dir /tmp/accelmark_test/
```

---

## Code Style Guidelines

- **No timing in platform scripts.** LoadGen owns all timing.
- **No hardcoded paths.** Use `_REPO_ROOT` from `runners/benchmark_runner.py`.
- **No per-request logging by default.** Suppress verbose framework logs unless `--verbose`.
- **Fail fast, fail clearly.** Raise exceptions with descriptive messages rather than returning None silently.
- **OOM is valid data.** Catch CUDA OOM in `inference_fn_offline`, raise a recognizable exception so LoadGen can record `"oom": true` and continue.

---

## Questions and Support

- **Bug in LoadGen or schema:** Open a GitHub Issue
- **New suite proposal:** Open a GitHub Issue with the "Request new suite" template
- **New platform support:** Open a PR with a working platform script and at least one verified result
- **Leaderboard question:** Check `leaderboard/generate.py` — it's well-commented

---

## Roadmap

The following features are planned but not yet implemented.
Contributors are welcome to pick these up — open an issue first to discuss.

### Leaderboard: implementation tab

The result modal currently has Details and Visualize tabs.
A third Implementation tab should show the runner's `meta.json` fields
(name, platform, framework, submitted_by, description, notes) and a
link to the runner folder on GitHub.

The `implementation_id` field is already present in new results.
The leaderboard generator (`leaderboard/generate.py`) already passes
`detail` and `viz` objects to the frontend — `impl` should follow the
same pattern via a new `extract_impl(result)` function.

The tab should only be visible when `row.implementation_id` is set.

### OpenAI-compatible serving API

Each runner already loads the model and exposes `inference_fn_streaming`.
A `--serve` mode on `run.py` would start a FastAPI server with
`/v1/chat/completions` and `/v1/models` endpoints backed by the same
inference stack that was benchmarked.

    python run.py --runner nvidia_vllm_e0859b3c --serve --port 8000

Since AccelMark has already measured performance for the runner, the
server could report capacity estimates (e.g. "Estimated: 5 max QPS
based on Suite A results") in the startup log.

Implementation notes:
- Add `serve()` method to `BenchmarkRunner` alongside `main()`
- Thin FastAPI adapter translating chat completions → `inference_fn_streaming`
- `run.py --serve` calls `serve()` instead of `main()`
- Optional `--workers N` for multi-request capacity