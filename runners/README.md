# AccelMark Runners

This directory contains all benchmark runners — both the official reference
implementations and community-submitted ones. Every runner is a self-contained
folder that you can run directly or submit to the leaderboard.

---

## What is a runner?

A runner is a Python class that wraps an inference framework (vLLM, LMDeploy,
TensorRT-LLM, mlx-lm, etc.) and exposes a standard interface. AccelMark's
benchmarking engine calls your runner to load the model and run inference —
everything else (timing, result building, Suite E scaling, accuracy gate) is
handled automatically.

You write ~50 lines. AccelMark handles the rest.

---

## Directory layout

```
runners/
├── benchmark_runner.py     ← Base class — inherit from this
├── protocol.py             ← RunnerProtocol — the serve layer interface
├── collect_env.py          ← Hardware/software detection
├── validate_submission.py  ← Result validator
├── validate_runners.py     ← Runner folder validator
├── hash_runner.py          ← Compute runner ID before submitting
├── meta.schema.json        ← JSON schema for meta.json
│
├── template/               ← starter template for new runners
│   └── runner.py
│
├── nvidia_vllm_{hash}/     ← Official reference runner (NVIDIA + vLLM)
│   ├── runner.py
│   ├── requirements.txt
│   └── meta.json
│
└── your_runner_{hash}/     ← Your runner goes here
    ├── runner.py
    ├── requirements.txt
    └── meta.json
```

---

## Quick start — 5 steps to get on the leaderboard

### Step 1 — Write your `runner.py`

Create a temporary folder and write your runner. It must inherit from
`BenchmarkRunner` and implement three methods:

> **Tip:** Copy `runners/template/runner.py` as your starting point.
> It has all required and optional methods pre-scaffolded with comments
> explaining each decision point.

```python
# runners/tmp/runner.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runners.benchmark_runner import BenchmarkRunner
from loadgen.types import InferenceResult
import time


class MyFrameworkRunner(BenchmarkRunner):

    SUPPORTS_STREAMING  = True    # set False if no token streaming API
    SUPPORTS_BATCHING   = True    # set False if serial only (e.g. mlx-lm)
    SUPPORTS_ONLINE     = True    # set False if no concurrent request support
    SUPPORTS_MULTI_CHIP = True    # set False if no tensor parallelism

    # SUPPORTED_PRECISIONS = what your framework can do on capable hardware.
    # BenchmarkRunner auto-detects hardware limits (V100→FP16, MI100→FP16, M1→FP16, etc.)
    # You almost never need to restrict this below ["bf16", "fp16", "fp32"].
    SUPPORTED_PRECISIONS = ["bf16", "fp16", "fp32"]

    # Declare supported quantization formats for Suite C.
    # BF16 is always included. List only formats your framework can load.
    # FP8 requires native FP8 hardware (H100, MI300X).
    SUPPORTED_QUANTIZATIONS = ["fp8", "w8a8", "w8a16", "w4a16"]  # H100 full support
    # SUPPORTED_QUANTIZATIONS = ["w8a8", "w8a16", "w4a16"]        # A100 (no FP8)
    # SUPPORTED_QUANTIZATIONS = ["w8a8", "w4a16"]                 # ROCm example
    # SUPPORTED_QUANTIZATIONS = []                                 # Apple MLX

    def load_model(self, model_path: str, suite: dict, tp_size: int) -> None:
        from myframework import Engine
        self.engine = Engine(model_path, tp_size=tp_size)

    def inference_fn_offline(self, prompts: list[str]) -> list[InferenceResult]:
        t_start = time.perf_counter()
        outputs = self.engine.generate(prompts)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return [
            InferenceResult(
                first_token_time_ms=None,
                total_time_ms=elapsed_ms,
                output_tokens=o.num_tokens,
                input_tokens=o.num_input_tokens,
                success=True,
                text=o.text,
            )
            for o in outputs
        ]

    async def inference_fn_streaming(self, prompt: str) -> InferenceResult:
        import time
        t_start = time.perf_counter()
        first_token_time_ms = None
        output_text = ""
        output_tokens = 0
        async for token in self.engine.stream(prompt):
            if first_token_time_ms is None:
                first_token_time_ms = (time.perf_counter() - t_start) * 1000
            output_text   += token
            output_tokens += 1
        return InferenceResult(
            first_token_time_ms=first_token_time_ms,
            total_time_ms=(time.perf_counter() - t_start) * 1000,
            output_tokens=output_tokens,
            input_tokens=0,
            success=True,
            text=output_text,
        )

    def release_resources(self) -> None:
        if self.engine is not None:
            del self.engine
            self.engine = None

    def _get_framework_name(self) -> str:
        return "MyFramework"

    def _get_framework_version(self) -> str:
        import myframework
        return myframework.__version__


if __name__ == "__main__":
    MyFrameworkRunner().main()
```

See the [reference implementation](nvidia_vllm_6e78e779/runner.py) and
[DEVELOPMENT.md](../DEVELOPMENT.md) for a full working example.

### Step 2 — Compute the hash and name your folder

The folder name encodes a content hash of `runner.py`. Compute it:

```bash
python runners/hash_runner.py runners/tmp/runner.py
# → nvidia_myframework_3f8a2c1d
```

Rename your folder to the printed ID:

```bash
mv runners/tmp runners/nvidia_myframework_3f8a2c1d
```

The hash changes every time `runner.py` changes. Once submitted, the folder
is **immutable** — updates require a new folder with a new hash.

### Step 3 — Write `meta.json`

```json
{
  "id":           "nvidia_myframework_3f8a2c1d",
  "platform":     "nvidia",
  "name":         "MyFramework on NVIDIA",
  "framework":    "MyFramework",
  "submitted_by": "your_github_username",
  "description":  "One sentence describing what makes this runner distinct.",
  "notes":        null,
  "created":      "2026-03-22"
}
```

`id` must exactly match the folder name.

### Step 4 — Validate

```bash
python runners/validate_runners.py runners/nvidia_myframework_3f8a2c1d/
```

Fix any errors before continuing. Warnings are OK.

### Step 5 — Submit a PR

Open a pull request adding your runner folder. The CI will re-run the
validator automatically and post a comment on the PR with the result.

---

## Folder naming convention

```
{platform}_{customname}_{hash8}
```

| Segment | Examples | Rules |
|---------|----------|-------|
| `platform` | `nvidia`, `amd`, `ascend`, `apple`, `other` | Lowercase, one of the allowed values |
| `customname` | `vllm`, `trtllm_fp8`, `lmdeploy_tp4` | Lowercase alphanumeric and underscores, your choice |
| `hash8` | `3f8a2c1d` | First 8 hex chars of SHA-256 of `runner.py` — computed automatically |

Examples:
```
nvidia_vllm_0ac7f5ba
nvidia_trtllm_fp8_8d2f1a4b
amd_vllm_rocm_7b2e1d8f
ascend_mindie_9c4a3f11
apple_mlx_b3e21f09
```

---

## Capability flags

Override these class attributes in your runner to declare what the framework supports:

| Flag | Default | Notes |
|------|---------|-------|
| `SUPPORTS_STREAMING` | `True` | Set `False` if no token streaming API — TTFT cannot be measured, online/interactive/sustained scenarios are skipped |
| `SUPPORTS_BATCHING` | `True` | Set `False` if serial only (e.g. mlx-lm) — offline runs one prompt at a time |
| `SUPPORTS_ONLINE` | `True` | Set `False` if framework cannot handle concurrent requests |
| `SUPPORTS_MULTI_CHIP` | `True` | Set `False` if no tensor parallelism — `--tensor-parallel-size` is ignored |
| `SUPPORTED_PRECISIONS` | `["bf16", "fp16", "fp32"]` | Maximum compute precisions on capable hardware. Hardware detection automatically restricts this (V100 → FP16, MI100 → FP16, M1 → FP16). Only restrict below the default if your framework genuinely cannot use a precision regardless of hardware. |
| `SUPPORTED_QUANTIZATIONS` | `[]` | Quantization formats supported for Suite C. Use uppercase strings: `"FP8"`, `"W8A8"`, `"W8A16"`, `"W4A16"`. BF16 is always supported and does not need to be listed. Empty list means this runner skips all quantized formats in Suite C. |

---

## Precision handling

AccelMark automatically resolves the correct compute precision for each run.
You rarely need to think about this — the defaults work for most runners.

**How it works:**

```
get_supported_precisions(chip_name, env_info) called
    │
    ├── Runner overrides it and returns a list?
    │   YES → use that list directly. Hardware detection skipped.
    │         (runner author's knowledge takes priority)
    │
    └── Runner returns None (default)?
            │
            ├── Tier 1: env_info has supports_bf16 field?
            │   (populated by collect_env.py for NVIDIA, AMD, Ascend, Apple)
            │
            ├── Tier 2: env_info has compute_capability?
            │   (NVIDIA fallback for older env_info.json files)
            │
            ├── Tier 3: chip name in known FP16-only list?
            │   (V100, T4, MI100, M1, etc.)
            │
            └── Default: assume BF16 capable
```

**The priority rule:**

| Runner says | Hardware detects | Result |
|---|---|---|
| YES to BF16 | NO (V100) | BF16 — trust runner |
| NO to BF16 | YES (A100) | FP16 — trust runner |
| Nothing (None) | NO (V100) | FP16 — hardware wins |
| Nothing (None) | YES (A100) | BF16 — hardware wins |

**Three runner personas:**

```python
# 1. Lazy (most common) — return None, auto-detection handles everything
def get_supported_precisions(self, chip_name, env_info):
    return None   # this is the default, you don't need to write it at all

# 2. Declarative — restrict via SUPPORTED_PRECISIONS, don't override the method
SUPPORTED_PRECISIONS = ["fp16", "fp32"]   # framework can't use BF16 at all

# 3. Expert — override for chip-specific knowledge hardware detection can't know
def get_supported_precisions(self, chip_name, env_info):
    # e.g. vLLM FP8 on H100 — framework-specific, not detectable from hardware info
    base = super().get_supported_precisions(chip_name, env_info)
    if "h100" in chip_name.lower():
        return (base or ["bf16", "fp16"]) + ["fp8"]
    return None   # auto-detect for other chips
```

When the run completes, `result.json` records the effective precision actually
used. The leaderboard shows an amber ⚠ when BF16 was requested but FP16 was used.

---

## Updating a runner

Runner folders are **immutable once merged**. Any edit to `runner.py` produces
a different hash, requiring a new folder. This ensures that `implementation_id`
in a result file always points to the exact code that produced it.

**Workflow for publishing an update:**

```bash
# 1. Edit runner.py in a temporary copy
cp -r runners/nvidia_vllm_0ac7f5ba runners/nvidia_vllm_tmp
# ... edit runners/nvidia_vllm_tmp/runner.py ...

# 2. Compute the new hash
python runners/hash_runner.py runners/nvidia_vllm_tmp/runner.py
# → nvidia_vllm_b3f29a11

# 3. Rename to the new hash
mv runners/nvidia_vllm_tmp runners/nvidia_vllm_b3f29a11

# 4. Update the new folder's meta.json
#    Set "id" to the new folder name
#    Add "supersedes": "nvidia_vllm_0ac7f5ba"

# 5. Add "deprecated_by" to the OLD folder's meta.json
#    "deprecated_by": "nvidia_vllm_b3f29a11"
#    "notes": "Deprecated — fixed edge case in release_resources()"

# 6. Validate the new runner
python runners/validate_runners.py runners/nvidia_vllm_b3f29a11/

# 7. Open a PR with both the new folder and the updated old meta.json
```

Old results remain valid — they still reference the original hash, which
still exists and is still runnable.

---

## Running a benchmark

Use `run.py` at the repo root to run any runner:

```bash
# List available runners
python run.py --list

# Run the default benchmark (offline + online + interactive + accuracy)
python run.py --runner nvidia_vllm_6e78e779 --suite suite_A

# Run a specific scenario only
python run.py --runner nvidia_vllm_6e78e779 --suite suite_A --scenario offline

# Run extra scenarios (e.g. sustained load test)
python run.py --runner nvidia_vllm_6e78e779 --suite suite_A --scenario sustained

# Run everything including extras
python run.py --runner nvidia_vllm_6e78e779 --suite suite_A --scenario all

# Multi-GPU
python run.py --runner nvidia_vllm_6e78e779 --suite suite_B --tensor-parallel-size 4

# Use a local model path
python run.py --runner nvidia_vllm_6e78e779 --suite suite_A --model-path /path/to/model
```

Results are written to `results/community/{chip}_{suite}_{runner_id}/`.
Running scenarios separately is safe — each goes into its own subdirectory
and the suite-level `result.json` is updated incrementally.

---

## Serving with your runner

Any runner that sets `SUPPORTS_STREAMING = True` can be used as an
OpenAI-compatible inference server:

```bash
pip install -r serve/requirements.txt

python run.py --runner nvidia_vllm_6e78e779 --serve --port 8000
```

See [serve/README.md](../serve/README.md) for full documentation.

---

## Further reading

- **[DEVELOPMENT.md](../DEVELOPMENT.md)** — full implementation
  guide with a complete working example (LMDeploy), capability flag details,
  platform detection, loadgen contract, and the runner update workflow
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** — how to submit
  benchmark results to the leaderboard
- **[serve/README.md](../serve/README.md)** — OpenAI-compatible serving API