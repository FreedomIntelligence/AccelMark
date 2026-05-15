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
├── CONTRIBUTING.md
├── DEVELOPMENT.md
├── runners/
│   ├── benchmark_runner.py ← Base class: all orchestration logic
│   ├── protocol.py         ← RunnerProtocol interface (serve layer)
│   ├── collect_env.py      ← Hardware/software detection
│   ├── validate_submission.py
│   ├── validate_runners.py ← CI: validate all runner folders
│   ├── hash_runner.py      ← Compute runner ID before submission
│   ├── meta.schema.json    ← JSON schema for runner meta.json
│   ├── template/runner.py  ← Annotated scaffold for new runners
│   └── nvidia_vllm_{hash}/ ← Reference runner (NVIDIA + vLLM)
│       ├── runner.py
│       ├── requirements.txt
│       └── meta.json
├── loadgen/
│   ├── loadgen.py          ← Shared timing and measurement engine
│   └── types.py            ← InferenceResult, SampleRecord
├── suites/
│   ├── suite_A/suite.json + requests.jsonl
│   ├── suite_B/suite.json + requests.jsonl
│   ├── suite_C/suite.json + suite.py + requests.jsonl
│   ├── suite_D/suite.json + requests.jsonl
│   ├── suite_E/suite.json + suite.py + requests.jsonl
│   ├── suite_F/suite.json + requests.jsonl
│   └── suite_G/suite.json + requests.jsonl
├── datasets/
│   ├── sharegpt_standard_v1/requests.jsonl  ← 500 prompts, ~280/310 tok
│   ├── sharegpt_longctx_v1/requests.jsonl   ← 200 prompts, ~28K input tok (Suite D)
│   └── sharegpt_edge_v1/requests.jsonl      ← 500 prompts, short-turn (Suite F)
├── serve/
│   ├── server.py           ← FastAPI OpenAI-compatible API
│   ├── adapter.py          ← Pydantic request/response models
│   └── tests/
├── schema/
│   ├── result.schema.json
│   ├── accuracy_subset.jsonl   ← immutable
│   └── accuracy_baselines.json ← MMLU baselines per model/precision
├── leaderboard/
│   ├── generate.py         ← reads results/, writes leaderboard.js + api/
│   └── site/
│       ├── index.html
│       └── api/            ← rank.json, chips.json, index.json, suites.json
├── results/
│   ├── verified/
│   └── community/
└── openclaw_skill/         ← Voice interface ("benchmark my GPU")
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
results/community/{run_name}/result.json   ← run_name is deterministic hash-based

GitHub Actions
    ↓  on push to results/
leaderboard/generate.py                    ← reads all result.json files
    ↓  writes
leaderboard/site/leaderboard.js
leaderboard/site/api/index.json            ← queried by OpenClaw Skill
```

**Output directory naming** — the output directory is named using `run_name`,
a deterministic string computed from the hardware + software + suite + submitter
config. Example:

```
results/community/nvidia_a100_sxm4_80gbx1_suite_A_nvidia_vllm_47f5d58e_ed4b0557
                  └──chip──────────────┘ └suite┘ └──runner──────────────┘ └run_id┘
```

The last 8 characters (`ed4b0557`) are the `run_id` — an 8-char hex hash that
uniquely identifies this configuration. See `_compute_run_id()` in
`benchmark_runner.py` for the hash inputs.

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
from runners.benchmark_runner import BenchmarkRunner, InferenceRequest
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
    def load_model(self, model_path: str, suite: dict, parallelism: dict) -> None:
        from lmdeploy import pipeline, TurbomindEngineConfig
        from transformers import AutoTokenizer

        tp_size = parallelism["tensor_parallel_size"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = pipeline(
            model_path,
            backend_config=TurbomindEngineConfig(tp=tp_size),
        )

    # ── Required: offline batch inference ────────────────────────────
    def inference_fn_offline(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        formatted = [self._format_prompt(r.prompt) for r in requests]
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

    # ── Optional: streaming for online/interactive/sustained ─────────
    async def inference_fn_streaming(self, request: InferenceRequest) -> InferenceResult:
        formatted = self._format_prompt(request.prompt)
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

    # ── Optional: runtime metrics (speculative decoding, MoE, etc.) ──
    def get_runtime_metrics(self) -> Optional[dict]:
        """Return framework-specific metrics after each inference run.

        Called once per scenario by the base class after `run()` completes.
        Return a dict with string keys and numeric/string values, or None.
        The result is stored verbatim in `task.runtime_metrics` of result.json
        and surfaced on the leaderboard.

        Example keys for speculative decoding:
          acceptance_rate (float 0-1), mean_accepted_tokens (float),
          draft_model_id (str)

        Example keys for MoE routing:
          expert_utilization_mean (float), load_balance_score (float)

        Base-class default returns None (no extra metrics collected).
        """
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

python run.py --runner nvidia_lmdeploy_{hash8} --suite suite_A
```

### Step 4: Test it

```bash
# Verify imports
python -c "from runners.nvidia_lmdeploy_{hash8}.runner import LMDeployRunner; print('OK')"

# Dry run
python run.py --runner nvidia_lmdeploy_{hash8} --help

# Full run
python run.py --runner nvidia_lmdeploy_{hash8} --suite suite_A
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

| Flag | Default | When to change |
|------|---------|----------------|
| `SUPPORTS_STREAMING` | `True` | Set `False` if framework has no token streaming API. TTFT will not be measured for online/interactive/sustained. |
| `SUPPORTS_BATCHING` | `True` | Set `False` if framework is serial only (e.g. mlx-lm). Offline runs requests one-by-one. |
| `SUPPORTS_ONLINE` | `True` | Set `False` if framework cannot handle concurrent requests. Online scenario is skipped. |
| `SUPPORTS_MULTI_CHIP` | `True` | Set `False` if no tensor parallelism. tensor_parallel_size from runner config and CLI is ignored. |
| `SUPPORTED_PRECISIONS` | `["bf16", "fp16", "fp32"]` | Maximum compute precisions on capable hardware. Hardware detection automatically restricts this at runtime. See *Precision resolution* below. |

### Example: Apple Silicon (no batching, no streaming)

```python
class MLXRunner(BenchmarkRunner):
    SUPPORTS_STREAMING = False   # mlx-lm has no streaming API
    SUPPORTS_BATCHING = False    # serial only
    SUPPORTS_MULTI_CHIP = False  # no tensor parallelism

    def load_model(self, model_path, suite, parallelism):
        from mlx_lm import load
        self.model, self.tokenizer = load(model_path)

    def inference_fn_offline(self, requests):
        # SUPPORTS_BATCHING=False: loadgen calls this one request at a time
        from mlx_lm import generate
        assert len(requests) == 1
        prompt = self._format_prompt(requests[0].prompt)
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

## Precision resolution

AccelMark automatically resolves the correct compute precision before each
model load. Understanding this is useful when adding support for hardware that
doesn't support BF16 (V100, T4, MI100, Apple M1, etc.).

### How it works

`BenchmarkRunner._resolve_precision(suite, env_info)` is called before every
`load_model()`. It uses a layered approach:

```
Step 1 — Ask the runner
    runner.get_supported_precisions(chip_name, env_info)
    Returns a list → use it directly, skip hardware detection
    Returns None   → proceed to step 2

Step 2 — Auto-detect from env_info (three tiers)
    Tier 1: env_info.accelerators[0].supports_bf16
            (set by collect_env.py for NVIDIA, AMD, Ascend, Apple)
    Tier 2: env_info.accelerators[0].compute_capability >= 8.0
            (NVIDIA fallback for older env_info.json files)
    Tier 3: chip name substring lookup
            (known FP16-only chips: v100, t4, mi100, m1, ...)
    Default: assume BF16 capable if nothing matches

Step 3 — Intersect with SUPPORTED_PRECISIONS
    (only applies when runner returns None)

Step 4 — Intersect with suite.allowed_precisions
    Fail with clear error if intersection is empty
```

### Priority rule

The runner always wins when it speaks. Hardware detection is only the fallback:

| Runner `get_supported_precisions` | Hardware detects | Resolved |
|---|---|---|
| Returns `["BF16", "FP16"]` | V100 (no BF16) | **BF16** — runner wins |
| Returns `["FP16"]` | A100 (has BF16) | **FP16** — runner wins |
| Returns `None` | V100 (no BF16) | **FP16** — hardware wins |
| Returns `None` | A100 (has BF16) | **BF16** — hardware wins |

### When to override `get_supported_precisions`

The default (`return None`) is correct for most runners — auto-detection handles
the common BF16/FP16 cases automatically.

Override when the runner has framework-specific knowledge hardware detection
cannot capture:

```python
def get_supported_precisions(self, chip_name: str, env_info: dict) -> list[str] | None:
    # vLLM FP8 is only useful on H100 — not detectable from hardware info alone
    base = super().get_supported_precisions(chip_name, env_info)
    if "h100" in chip_name.lower():
        return (base or ["bf16", "fp16"]) + ["fp8"]
    return None   # auto-detect for all other chips

# Framework has a BF16 bug on a specific chip
def get_supported_precisions(self, chip_name: str, env_info: dict) -> list[str] | None:
    if "a100" in chip_name.lower():
        return ["fp16", "fp32"]   # force FP16 even though A100 supports BF16
    return None
```

Returning `None` from a chip-specific branch means auto-detection handles
that chip — you only need to cover cases where your knowledge differs from
hardware capability.

### `SUPPORTED_PRECISIONS` vs `get_supported_precisions`

Use `SUPPORTED_PRECISIONS` when the restriction applies to **all hardware**:

```python
# Framework genuinely cannot use BF16 on any hardware
SUPPORTED_PRECISIONS = ["fp16", "fp32"]
```

Use `get_supported_precisions()` when the restriction or addition is
**chip-specific**:

```python
# FP8 only on H100, auto-detect everything else
def get_supported_precisions(self, chip_name, env_info):
    base = super().get_supported_precisions(chip_name, env_info)
    if "h100" in chip_name.lower():
        return (base or []) + ["fp8"]
    return None
```

### Hardware detection in `collect_env.py`

`collect_env.py` populates `supports_bf16` on each accelerator entry:

| Platform | Detection method |
|---|---|
| NVIDIA | `compute_capability >= 8.0` (V100=7.0, T4=7.5, A100=8.0, H100=9.0) |
| AMD | `gfx` architecture code (gfx908/MI100=no, gfx90a/MI250X=yes, gfx942/MI300X=yes) |
| Ascend | Chip model name (910B=yes, 310=no) |
| Apple | Chip generation (M1=no, M2/M3/M4=yes) |

When adding a new platform, populate `supports_bf16` in your `collect_*()` function.
See the existing collectors for reference.

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
   e.g. quantization format (BF16 / FP8 / W8A8 / W8A16 / W4A16)

3. What model?
   Use a model that is already in another suite if possible.
   New models require downloading and generating new requests.jsonl.

4. What scenarios?
   offline: always include (throughput is the most comparable metric)
   online: include if latency under load matters
   interactive: include if single-user latency matters
   sustained: include as an extra if long-run stability matters
   speculative: include as an extra if the suite targets compute-bound acceleration (draft model fields required: speculative_draft_model_id, speculative_draft_model_revision, speculative_num_tokens)
   burst: include as an extra if the suite tests bursty traffic patterns (burst fields required: burst_steady_qps, burst_peak_qps, burst_duration_seconds, burst_interval_seconds); requires SUPPORTS_STREAMING = True in runner

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
  "dataset": "sharegpt_standard_v1",
  "scenarios": {
    "default": ["accuracy", "offline", "online", "interactive"],
    "extra":   ["sustained"]
  },
  "precision_required": "BF16",
  "allowed_precisions": ["BF16", "FP16"],
  "request_distribution": {
    "input_tokens_p50": 280,
    "output_tokens_p50": 310,
    "source": "shibing624/sharegpt_gpt4"
  },
  "output_tokens_max": 512,
  "concurrency_levels": [8, 32, 128],
  "online_qps_levels": [5, 25, 100],
  "online_sla_ttft_ms": 500,
  "num_runs": 3,
  "warmup_runs": 1,
  "online_warmup_runs": 0,
  "interactive_warmup_runs": 0,
  "accuracy_threshold_delta": 0.1,
  "request_count": 200,
  "online_request_count": 500,
  "interactive_request_count": 100,

  "_comment_speculative": "Optional — add to scenarios.extra when testing speculative decoding",
  "speculative_draft_model_id": "meta-llama/Llama-3.2-1B-Instruct",
  "speculative_draft_model_revision": "<commit-sha>",
  "speculative_num_tokens": 4,

  "_comment_burst": "Optional — add to scenarios.extra when testing burst load (requires SUPPORTS_STREAMING = True)",
  "burst_steady_qps": 5,
  "burst_peak_qps": 25,
  "burst_duration_seconds": 30,
  "burst_interval_seconds": 120
}
```

### Step 3: Choose or create a dataset

If your suite uses a standard prompt distribution, reference an existing
shared dataset:

```json
"dataset": "sharegpt_standard_v1"
```

Available datasets are in `datasets/`. Check `datasets/README.md` for
descriptions and distributions.

If you need a custom distribution:

1. Create `datasets/{your_dataset}_v1/requests.jsonl`
2. Create `datasets/{your_dataset}_v1/README.md`
3. Set `"dataset": "{your_dataset}_v1"` in your suite.json

If your suite needs a custom dataset only used by that suite, you can
also place `requests.jsonl` directly in `suites/suite_X/` — the
benchmark runner checks there as a fallback.

Dataset format (one JSON object per line):
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
python run.py --runner nvidia_vllm_47f5d58e \
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

## Suite plugin system

Suites with custom orchestration logic (multiple subprocesses, special
merge logic) can provide a `suite.py` file in their folder.
`BenchmarkRunner.main()` checks for this file and delegates to it when
present. Suites without a `suite.py` use the generic scenario dispatch.

### When to use `suite.py`

Use it when your suite needs orchestration that `_run_all_scenarios()`
cannot handle generically:

- **Multiple subprocesses in sequence** — Suite C runs one subprocess
  per precision format; Suite E runs one per chip count
- **Custom merge logic** — combining results from subprocesses into a
  single suite-level `result.json` with derived metrics
- **Non-standard scenario ordering** — e.g. accuracy must run before
  other scenarios as a gate

Suites that run standard scenarios (offline, online, interactive,
sustained) on a single model do NOT need `suite.py`.

### File structure

```
suites/
├── suite_A/
│   ├── suite.json        ← no suite.py needed
│   └── requests.jsonl
├── suite_B/
│   ├── suite.json
│   └── requests.jsonl
├── suite_C/
│   ├── suite.json
│   ├── suite.py          ← custom quantization orchestration
│   └── requests.jsonl
├── suite_D/
│   ├── suite.json
│   └── requests.jsonl    ← long-context; no suite.py
├── suite_E/
│   ├── suite.json
│   ├── suite.py          ← custom scaling orchestration
│   └── requests.jsonl
└── suite_F/
    ├── suite.json
    └── requests.jsonl    ← consumer/edge; no suite.py
```

### Required interface

`suite.py` must export a single `run()` function:

```python
def run(br, args, suite: dict, env_info: dict) -> None:
    """
    Suite entry point called by BenchmarkRunner.main().

    Args:
        br:       BenchmarkRunner instance — full access to all methods
        args:     Parsed argparse.Namespace from parse_args()
        suite:    Parsed suite.json dict
        env_info: Hardware/software info from collect_env.py
    """
```

The `br` parameter gives full access to all BenchmarkRunner methods:
`br._run_single_scenario()`, `br._merge_scenario_results()`,
`br._resolve_model_path()`, `br._build_result_json()`, etc.

### Delegating single-scenario runs

`suite.py` typically only handles `--scenario default` and `--scenario all`.
For single scenarios (e.g. `--scenario offline`), delegate back to the
base class:

```python
def run(br, args, suite, env_info):
    if args.scenario in ("default", "all"):
        _run_my_suite(br, args, suite)
    else:
        br._setup_logging(args.output_dir)
        br._run_single_scenario(args, suite)
```

### Using base class methods

Common patterns:

```python
# Resolve model path (checks models_local.yaml)
path = br._resolve_model_path(model_id, args.model_path)

# Parse scenarios config — expects {"default": [...], "extra": [...]}
default, extra = br._parse_scenarios_config(suite)

# Merge scenario results after running offline+online+interactive
br._merge_scenario_results(base_dir, suite, successful, elapsed)

# Run a single scenario as subprocess
# (use sys.argv[0] as the platform script path)
```

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
                "supports_bf16": True,   # set based on chip model/generation
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
def inference_fn(requests: list[InferenceRequest]) -> list[InferenceResult]:
    # Must return one InferenceResult per request (same order)
    # Read request.prompt for the formatted prompt string
    # Do NOT time anything — LoadGen handles all timing
    prompts = [r.prompt for r in requests]
    ...
```

**Online, interactive, and sustained scenarios** (async):
```python
async def inference_fn(request: InferenceRequest) -> InferenceResult:
    # Must be a coroutine (async def)
    # LoadGen schedules concurrent calls for online/sustained
    # LoadGen awaits serially for interactive
    # first_token_time_ms should be set if streaming is available
    formatted = self.format_prompt(request.prompt)
    ...
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
    output_text: Optional[str] = None     # generated text (used by accuracy scoring and serve layer)
```

### What LoadGen measures

| Scenario | Measures | Primary metric |
|----------|----------|----------------|
| offline | Total tokens / elapsed time | `throughput_tokens_per_sec` (input + output) |
| online | TTFT distribution at each QPS level | `max_valid_qps` (highest QPS with p99 TTFT < SLA) |
| interactive | TTFT distribution, serial requests | `ttft_ms_p99` |
| sustained | Throughput + TTFT sampled every N seconds over 30 min | `sustained_throughput_tokens_per_sec`, `throttle_ratio` |
| speculative | Offline throughput with draft model (same path as offline, engine uses speculative decoding) | `throughput_tokens_per_sec`; optional `task.runtime_metrics.acceptance_rate` if runner overrides `get_runtime_metrics()` |
| burst | Two-state bursty load: alternates steady QPS and burst QPS windows | `burst_degradation_ratio` (burst_ttft_p99 / steady_ttft_p99); `sla_met_during_burst` |

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

**`meta` fields for run identity and status:**

| Field | Type | Description |
|-------|------|-------------|
| `meta.run_id` | string\|null | 8-char hex hash of hardware+software+suite+submitter. Deterministic — same config always produces same `run_id`. Used for duplicate detection. |
| `meta.run_name` | string\|null | Full directory name: `{chip}x{count}_{suite}_{runner}_{run_id}`. Used as the output directory name. |
| `meta.time` | string\|null | Benchmark start time HH:MM:SS. |
| `meta.flagged` | string\|null | Null for normal results. Set to a reason string if community review concludes the result is suspicious (via a follow-up PR) — triggers ⚠️ badge on leaderboard. |

These fields are optional in the schema for backward compatibility with older results.
New benchmark runs populate all four automatically.

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
python runners/validate_runners.py runners/nvidia_vllm_47f5d58e/
```

This validates a single runner folder and tells you clearly whether it is
ready to submit:

```
Validating: nvidia_vllm_47f5d58e/
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
✓ PASSED — nvidia_vllm_47f5d58e is ready to submit
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
| `deprecated_by` target exists | Warning if the referenced new runner folder is not found |

`requirements.txt` absence is a warning, not an error.
`deprecated_by` cross-reference failures are also warnings —
the referenced folder may not be merged yet when validating locally.

**Hash mismatch** is the most common failure after editing `runner.py` without
renaming the folder. The error message tells you exactly what to do:

```
  ✗ Hash mismatch.
      Folder ends with : e0859b3c
      runner.py hashes to: 6e78e779
      Rename folder to: nvidia_vllm_47f5d58e
```

To compute the correct name before creating a new runner folder:

```bash
python runners/hash_runner.py path/to/your/runner.py
# → nvidia_vllm_47f5d58e
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
# → nvidia_vllm_47f5d58e
```

**Step 2: Create the new runner folder**

```bash
cp -r runners/nvidia_vllm_old_hash runners/nvidia_vllm_47f5d58e
# Apply your edits to runners/nvidia_vllm_47f5d58e/runner.py
```

**Step 3: Update `meta.json` in the new folder**

```json
{
  "id":           "nvidia_vllm_47f5d58e",
  "platform":     "nvidia",
  "name":         "vLLM on NVIDIA (reference implementation)",
  "framework":    "vLLM",
  "submitted_by": "JuhaoLiang1997",
  "description":  "...",
  "supersedes_chain": ["nvidia_vllm_old_hash"],
  "notes":        null,
  "created":      "YYYY-MM-DD"
}
```

**Step 4: Add `deprecated_by` to the old runner's `meta.json`**

`meta.json` is the only file that may be edited in an existing runner folder.

```json
{
  "id":            "nvidia_vllm_old_hash",
  "deprecated_by": "nvidia_vllm_47f5d58e",
  "notes":         "Deprecated — use nvidia_vllm_47f5d58e. Fixed edge case in release_resources()."
}
```

**Step 5: Validate and submit**

```bash
python runners/validate_runners.py runners/nvidia_vllm_47f5d58e/
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
python run.py --runner nvidia_vllm_47f5d58e \
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