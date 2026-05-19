---
name: Propose a new suite
about: Propose a new benchmark suite (new model, scenario mix, or scaling axis)
title: "[Suite] <short description, e.g. 'Suite H — Llama-3.1-405B'>"
labels: suite-proposal
assignees: ''
---

<!--
  This template starts the discussion for a new AccelMark suite. The final
  contract goes into suites/<suite_id>/suite.json (see
  schema/suite.schema.json) — please fill in as many of the fields below as
  you can. Anything you leave blank we'll work out in the thread before
  merging.

  Full walk-through: DEVELOPMENT.md "Adding a new suite"
                     https://github.com/FreedomIntelligence/AccelMark/blob/main/DEVELOPMENT.md
-->

## Why this suite?

<!-- One sentence: the question this suite answers that no existing suite
     (A–G) covers. Example: "How fast is this chip on 405B-parameter
     dense models?" -->

## Suite contract (draft)

| Field | Proposed value |
|---|---|
| **Suite ID** | `suite_<X>` |
| **Model** | `<huggingface/repo-id>` |
| **Model revision** | `<commit sha or tag>` |
| **Chip count** | `1` / `auto` / specific number |
| **Precision** | `BF16` / `FP16` / list of allowed precisions |
| **Dataset** | existing (`sharegpt_standard_v1`, `sharegpt_edge_v1`, `sharegpt_longctx_v1`) or new |
| **Max model length** | tokens |
| **Output tokens (max)** | tokens |
| **Concurrency levels** | e.g. `[8, 32, 128]` |
| **Default scenarios** | subset of `accuracy / offline / online / interactive / sustained` |
| **Extra scenarios** | optional: `sustained / speculative / burst / …` |
| **Primary metric** | `offline_throughput`, `max_valid_qps`, … |
| **Expected run time on A100** | minutes |

## Accuracy baseline

<!-- Required before the suite can land on the main leaderboard. -->

- [ ] I will provide an A100 (or equivalent reference) BF16 baseline score
      to add to `schema/accuracy_baselines.json`.
- [ ] If a new dataset is required, I will submit it under
      `datasets/<name>_v1/` with a `README.md` that documents the source
      and upstream license (see [`datasets/README.md`](../../datasets/README.md)).

## Custom orchestration?

<!-- Most suites only need `suite.json`. Mark these only if you genuinely
     need a `suite.py` plugin (multiple subprocesses, custom merge logic,
     similar to Suite C/E). -->

- [ ] Standard scenario dispatch is enough — no `suite.py` needed.
- [ ] A `suite.py` plugin is required. Reason:

## Reference result plan

<!-- New suites do not appear on the main leaderboard until at least one
     verified reference result is submitted. -->

- Reference hardware: <e.g. NVIDIA A100-SXM4-80GB ×1>
- Runner: `<runner_id>`
- Who will run it: <@your-handle / vendor / community member>

## Open questions

<!-- Anything you'd like community / maintainer feedback on before opening
     the PR. -->
