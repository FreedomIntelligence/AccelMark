# Zero-GPU Advisor & Static JSON API

You don't need a GPU to use AccelMark data in scripts, procurement sheets, notebooks, or AI agents.

## 1. Static JSON API (Auth-Free)

Every merge to `main` publishes static JSON files on GitHub Pages under `https://freedomintelligence.github.io/AccelMark/api/`:

| Endpoint | Description |
|---|---|
| [`api/manifest.json`](https://freedomintelligence.github.io/AccelMark/api/manifest.json) | Snapshot version, timestamp, counts, and SHA-256 checksums for all files |
| [`api/chips.json`](https://freedomintelligence.github.io/AccelMark/api/chips.json) | Indexed accelerators, submission counts, best & median throughput |
| [`api/index.json`](https://freedomintelligence.github.io/AccelMark/api/index.json) | Per-chip suite breakdown for fast programmatic lookup |
| [`api/rank.json`](https://freedomintelligence.github.io/AccelMark/api/rank.json) | Per-submission suite ranks, framework, runner tier, and primary metric |
| [`api/suites.json`](https://freedomintelligence.github.io/AccelMark/api/suites.json) | Suite specifications, models, and SLA parameters |
| [`api/schema.json`](https://freedomintelligence.github.io/AccelMark/api/schema.json) | JSON Schema for `result.json` submissions |

## 2. Local CLI Advisor (`tools/advisor.py`)

```bash
# Best Tok/$ options under $2.00/hr using Closed reference runners
python tools/advisor.py --suite suite_A --budget 2.0 --tier reference

# Compare frameworks (vLLM vs SGLang vs TensorRT-LLM) on the same chip
python tools/advisor.py --engine-compare "A100"

# Export a Markdown or JSON comparison table
python tools/advisor.py --compare "4090" "3090" "A100" --format markdown
```
