# Platform Script Template

Use `run_benchmark.py` as a starting point for adding a new platform.

## Steps

1. Copy `run_benchmark.py` to `scripts/{your_platform}/run_{framework}.py`
2. Implement the three TODO sections:
   - `load_model()` — initialize your platform's model
   - `inference_fn()` — run inference with streaming, return `InferenceResult` objects
   - `get_peak_memory_gb()` — return peak memory usage
3. Add `scripts/{your_platform}/README.md` with setup instructions
4. Add `scripts/{your_platform}/requirements.txt`
5. Test by running Suite A offline and validating with `scripts/validate_submission.py`

## Key Rules

- **Do not modify `loadgen/`** — all timing is handled there
- **Use streaming output** to measure TTFT accurately
- **Return results in the same order as inputs** from `inference_fn`
- **Load the model once** in `load_model()`, not on every call to `inference_fn`
