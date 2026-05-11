# AccelMark Datasets

Shared request datasets used across benchmark suites.

| Dataset | Suites |
|---------|--------|
| `sharegpt_standard_v1` | A, B, C, E, G |
| `sharegpt_longctx_v1` | D |
| `sharegpt_edge_v1` | F |

Each folder has its own `README.md` with token statistics and source notes.

## Directory naming

```
{source}_{variant}_v{N}/
```

Dataset folders are **immutable once merged** — any change in prompts or
distribution creates a new versioned folder (`v2`, `v3`, etc.).
Suites reference a dataset by exact folder name, so changing a dataset
never silently affects existing results.

## Adding a dataset

1. Create `datasets/{name}_v1/requests.jsonl`
2. Create `datasets/{name}_v1/README.md` describing the source and distribution
3. Reference it from your suite: `"dataset": "{name}_v1"` in suite.json
4. Submit a PR

## Format

Each line in `requests.jsonl`:
```json
{
  "request_id": 0,
  "prompt": "...",
  "input_tokens": 245,
  "conversation_id": "sg_00001",
  "turn_index": 0,
  "prompt_type": "conversational"
}
```
