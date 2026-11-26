# Reproduce & Verify a Result (±5%)

AccelMark's two-tier trust model promotes `community` submissions to **`verified`** once an independent submitter reproduces the measurement within **±5%** on matching hardware and runner.

## 1. Pick an Open Quest

Browse [**Reproduction Quests ↗**](https://freedomintelligence.github.io/AccelMark/#/reproduce) and filter by your hardware vendor or suite. Click **Copy rerun** on any row to copy the exact reproduction command.

## 2. Run the Reproduction

```bash
python run.py --runner <same_runner_folder> \
  --suite <suite_id> \
  --reproduces-run-id <original_run_id> \
  --tier verified
```

The harness records `meta.reproduces_run_id` inside your `result.json` so the leaderboard automatically links both submissions and credits both the original author and the verifier.

## 3. Submit Your Verification PR

```bash
python run.py --submit results/community/<your_run_dir>
```

Once merged, if the primary metric agrees within **±5%**, both entries display the green **Verified** shield on the leaderboard.
