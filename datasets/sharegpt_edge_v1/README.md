# sharegpt_edge_v1

Short-turn ShareGPT conversational prompts. Used by Suite F (consumer/edge benchmark).

Filtered from `shibing624/sharegpt_gpt4` to retain only short-turn exchanges,
producing a distribution representative of interactive consumer inference workloads.

| Field             | Value                            |
|-------------------|----------------------------------|
| Source            | shibing624/sharegpt_gpt4         |
| Prompts           | 500                              |
| Input tokens p50  | ~95                              |
| Input tokens p99  | ~600                             |
| Output tokens p50 | ~150                             |
| Output tokens p99 | ~400                             |
| Type              | Conversational, single-turn      |

## Difference from sharegpt_standard_v1

`sharegpt_standard_v1` (Suites A, B, C, and E) has p50 input ~280 tokens and p50 output ~310 tokens.
`sharegpt_edge_v1` uses shorter prompts to keep benchmark runtime practical on consumer GPUs
and to reflect the latency-sensitive interactive use cases they are typically deployed for.