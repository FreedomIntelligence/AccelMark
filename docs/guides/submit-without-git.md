# Submit Without Git (`--zip`)

Don't want to fork, branch, or use `git`? Package your validated result into a single `.zip` archive and upload it through our GitHub Issue Form.

## 1. Validate & Package

```bash
python run.py --submit results/community/<run_dir> --zip
```

This runs the exact same schema and protocol checks as CI and writes `<run_dir>.zip` containing `result.json` and `env_info.json`.

## 2. Upload via GitHub Issue

1. Open the [**Result Submission Issue Form**](https://github.com/FreedomIntelligence/AccelMark/issues/new?template=result_submission.yml).
2. Fill in your accelerator model, suite, and runner ID.
3. Drag and drop the `.zip` file into the description box and click **Submit new issue**.

A maintainer will run CI validation and merge your result into the leaderboard with full submitter attribution.
