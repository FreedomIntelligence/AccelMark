# Apple Silicon — mlx-lm runner

## Requirements

- macOS on **Apple Silicon** (M1/M2/M3/M4…)
- Python 3.10+
- `pip install -r requirements.txt` (from this folder)

## Quick smoke test

From the AccelMark repo root:

```bash
python runners/apple_mlx_lm_9546b8b5/smoke_test.py Qwen2.5-0.5B-Instruct-bf16
```

Use a local model directory or a Hugging Face id that `mlx_lm.load()` accepts.

## Benchmark

```bash
python run.py --runner apple_mlx_lm_9546b8b5 --suite suite_A --scenario offline
```

`--scenario online` is not supported (runner sets `SUPPORTS_ONLINE = False`).

## Implementation notes

- Mirrors `mlx_example.py`: `mlx_lm.load` + `stream_generate` / token streaming.
- Offline throughput: one `stream_generate` pass per request; all results share the batch wall-clock `total_time_ms` (same convention as vLLM).
