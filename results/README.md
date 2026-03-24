# Results

Benchmark results organized by trust tier.

## Tiers

| Tier | Description |
|------|-------------|
| `verified/` | Independently reproduced by a maintainer within 5%. Shown on the main leaderboard. |
| `community/` | Passed schema validation, not yet independently reproduced. Shown on the community tab. |

Submissions start in `community/` and may be promoted to `verified/` by a maintainer.

---

## Directory naming

```
{chip}_{suite}_{runner_id}
```

| Segment | Example | Description |
|---------|---------|-------------|
| `chip` | `nvidia_a100sxm480gbx1` | Chip name + memory + chip count |
| `suite` | `suite_A` | Suite ID — matches `suite_id` in `result.json` |
| `runner_id` | `nvidia_vllm_0ac7f5ba` | Full runner folder name including hash |

**Example:**
```
results/community/nvidia_a100sxm480gbx1_suite_A_nvidia_vllm_0ac7f5ba/
```

If no runner ID is available,
the runner segment is `unknown_{random8}`:
```
results/community/nvidia_a100sxm480gbx1_suite_A_unknown_3f8a2c1d/
```

---

## Submission directory structure

Each submission directory contains per-scenario subdirectories and a
merged suite-level result at the top level:

```
nvidia_a100sxm480gbx1_suite_A_nvidia_vllm_0ac7f5ba/
├── result.json              ← Merged suite-level result — primary submission artifact
├── env_info.json            ← Hardware/software environment
├── accuracy/
│   ├── result.json
│   └── accuracy.json        ← Accuracy gate result
├── offline/
│   └── result.json
├── online/
│   └── result.json
├── interactive/
│   └── result.json
└── sustained/               ← Present only if --scenario sustained was run
    └── result.json
```

`result.json` at the top level is updated incrementally — running scenarios
separately produces the same final result as running them all at once.

Files not tracked in git (stay on the submitter's machine):
- `run.log` — full benchmark log
- `*/samples.jsonl` — per-request raw samples
- `accuracy/accuracy_outputs.jsonl` — per-question accuracy outputs

---

## Submitting

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full guide.

To submit a runner (new inference framework), see [runners/README.md](../runners/README.md).