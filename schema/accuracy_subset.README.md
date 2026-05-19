# `accuracy_subset.jsonl` — accuracy gate question bank

100 multiple-choice items drawn from
[MMLU](https://github.com/hendrycks/test) (Massive Multitask Language
Understanding). Every benchmark run executes this subset against the loaded
model as a "model-quality sanity check" before measuring throughput or
latency. The subset is **immutable** — see `CONTRIBUTING.md` "A few rules"
and `benchmark_runner.py::_run_accuracy_scenario`.

## File format

One JSON object per line:

```json
{
  "question_id": "mmlu_0096",
  "subject": "machine_learning",
  "question": "Which of the following statements about Naive Bayes is incorrect?",
  "choices": ["...", "...", "...", "..."],
  "answer": "B"
}
```

| Field         | Notes                                             |
|---------------|---------------------------------------------------|
| `question_id` | Stable identifier (`mmlu_<index>`) — never reused |
| `subject`     | MMLU subject tag (e.g. `machine_learning`)        |
| `question`    | Plain-text prompt                                 |
| `choices`     | List of exactly 4 strings                         |
| `answer`      | Letter in `{"A", "B", "C", "D"}`                  |

## How AccelMark uses it

- Loaded by `runners/benchmark_runner.py` (`_run_accuracy_scenario`, ~line 1700).
- Scored as `correct / total`; compared against per-suite baselines in
  [`accuracy_baselines.json`](accuracy_baselines.json).
- A failed gate aborts the benchmark unless the user passes
  `--skip-accuracy-gate` (the resulting submission is permanently flagged).

This is **not** a measurement of MMLU performance — the subset is too small.
It exists only to catch grossly broken model weights / quantization configs
before runtime measurements waste hours of compute.

## License & attribution

The questions are a 100-item subset of MMLU:

> Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., &
> Steinhardt, J. (2021). **Measuring Massive Multitask Language
> Understanding.** *International Conference on Learning Representations.*
> arXiv:[2009.03300](https://arxiv.org/abs/2009.03300)
> Source: <https://github.com/hendrycks/test>

MMLU is distributed under the **MIT License**. AccelMark redistributes
this subset under the same license; the AccelMark Apache-2.0 license
covers only the surrounding evaluation code, not the question content.

See [`../NOTICE`](../NOTICE) for the full third-party attribution.
