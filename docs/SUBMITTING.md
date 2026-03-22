# Submitting Results to AccelMark Leaderboard

## Prerequisites

Before submitting, ensure:
- [ ] Benchmark completed without errors
- [ ] `configs/submitter.yaml` has your GitHub username
- [ ] Accuracy check completed (or reused from previous run)
- [ ] Result passes validation

---

## Step 1: Run accuracy check (if not auto-reused)

```bash
python scripts/run_accuracy.py \
    --model-path /path/to/model \
    --suite suite_A \
    --output results/community/my_submission/accuracy.json
```

If accuracy was auto-reused from a previous run, this step is skipped automatically.

---

## Step 2: Validate your submission

```bash
python scripts/validate_submission.py \
    --dir results/community/my_submission/
```

Fix any errors before proceeding. Common issues:
- `submitted_by` is empty → fill in `configs/submitter.yaml`
- `accuracy.valid` is false → rerun accuracy check
- `throughput` is 0 → benchmark failed, check run.log

---

## Step 3: Submit via GitHub Issue

Go to: https://github.com/JuhaoLiang1997/AccelMark/issues/new

Select template: **Community Submission**

Paste your `result.json` content into the issue body.

The CI bot will:
1. Validate the submission automatically
2. Create a PR to add your result to `results/community/`
3. Update the leaderboard within minutes

---

## Submission tiers

| Tier | Requirements | Leaderboard |
|------|-------------|-------------|
| `community` | Self-reported, passes schema validation | Community tab |
| `verified` | Reproduced by maintainers within 5% | Main leaderboard |

Community results appear immediately after CI validation.
Verified status is granted after manual reproduction.

---

## What makes a good submission

- Run on dedicated hardware (no other GPU workloads)
- Use the exact model revision specified in suite.json
- Include env_info.json (auto-generated)
- All three scenarios completed (offline + online + interactive)
- No `--enforce-eager` unless unavoidable (note it in meta.notes)
