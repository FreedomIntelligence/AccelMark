# sharegpt_longctx_v1

Long-context prompts for Suite D (~28K-token inputs; `max_model_len` 30,208 in `suite_D/suite.json`).

| Field | Value |
|---|---|
| Source | [shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) (long-context subset) |
| Prompts | 200 |
| Input tokens p50 | ~28,000 |
| Output tokens p50 | ~256 (suite caps generation) |
| Type | Document QA, long-form input |

## License & attribution

The prompts are derived from the same ShareGPT GPT-4 corpus as
`sharegpt_standard_v1` and are redistributed under the upstream license,
**[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**. Long-context
items are selected by tokenized input length; no additional editorial
modification beyond filtering is applied.

Apache-2.0 (the AccelMark repository license) covers only the AccelMark code,
schemas, and selection logic — not the prompt text itself. See [`../../NOTICE`](../../NOTICE)
for the full third-party attribution.

If you use these prompts in research, please cite the upstream dataset and
this repository.
