# ShareGPT SWA Dataset v1

Long-context benchmark dataset for Suite H (Sliding Window Attention).

## Source
Prompts from `sharegpt_longctx_v1`, truncated to the last 10,000 tokens of each conversation. This ensures prompts exceed the sliding window (4096 tokens for Mistral-7B) while staying within the model's maximum context (32,768 tokens).

## Purpose
Suite H uses Mistral-7B-Instruct-v0.1 (`sliding_window=4096`) to test how Sliding Window Attention changes a model's position on the arithmetic intensity spectrum. Prompts must be longer than the sliding window to trigger SWA-specific behavior in the attention computation and KV cache management.

## Statistics
- 100 prompts
- Input tokens: 10,000 (all prompts truncated uniformly)
- Source: sharegpt_longctx_v1 (original prompts 24K-33K tokens)

## Framework Compatibility Note
Whether the inference framework honors `sliding_window` for KV cache allocation is framework- and version-dependent. This dataset provides the prompts needed to test SWA; the framework's SWA support determines whether the benchmark measures SWA-specific effects or dense-equivalent behavior.
