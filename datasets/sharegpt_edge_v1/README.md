# sharegpt_edge_v1

Short-turn ShareGPT conversational prompts. Used by Suite F (consumer/edge benchmark).

Filtered from `shibing624/sharegpt_gpt4` to retain only short-turn exchanges,
producing a distribution representative of interactive consumer inference workloads.

| Field             | Value                                                                                |
|-------------------|--------------------------------------------------------------------------------------|
| Source            | [shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) |
| Prompts           | 500                                                                                  |
| Input tokens p50  | ~95                                                                                  |
| Input tokens p99  | ~600                                                                                 |
| Output tokens p50 | ~150                                                                                 |
| Output tokens p99 | ~400                                                                                 |
| Type              | Conversational, single-turn                                                          |

## Difference from sharegpt_standard_v1

`sharegpt_standard_v1` (Suites A, B, C, and E) has p50 input ~280 tokens and p50 output ~310 tokens.
`sharegpt_edge_v1` uses shorter prompts to keep benchmark runtime practical on consumer GPUs
and to reflect the latency-sensitive interactive use cases they are typically deployed for.

## License & attribution

The prompts are derived from `shibing624/sharegpt_gpt4` and are redistributed
under the upstream license, **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.

Apache-2.0 (the AccelMark repository license) covers only the AccelMark code,
schemas, and selection logic — not the prompt text itself. See [`../../NOTICE`](../../NOTICE)
for the full third-party attribution.

If you use these prompts in research, please cite the upstream dataset and
this repository.