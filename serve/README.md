# AccelMark Serve

Run any AccelMark runner as an OpenAI-compatible inference server.

The same code that produced your benchmark result serves your API —
no re-configuration, no separate deployment stack.

---

## Quick start

```bash
# Install serve dependencies
pip install -r serve/requirements.txt

# Option A — use a benchmark suite (model + params come from suite.json)
python run.py --runner nvidia_vllm_bc2ddb31 --suite suite_A --serve

# Option B — specify the model directly, no suite required
python run.py --runner nvidia_vllm_bc2ddb31 --model meta-llama/Llama-3.1-8B-Instruct --serve

# Test it
curl http://localhost:8000/health
```

---

## Installation

The serve layer has its own dependencies separate from benchmark runners:

```bash
pip install -r serve/requirements.txt
```

You also need the runner's own dependencies installed:

```bash
pip install -r runners/nvidia_vllm_bc2ddb31/requirements.txt
```

---

## Starting the server

There are two ways to specify the model and generation parameters:

### Option A — suite-based (benchmark config as source of truth)

```bash
python run.py --runner nvidia_vllm_bc2ddb31 --suite suite_A --serve
```

The suite's `model_id`, `output_tokens_max`, and `max_model_len` are used automatically.

### Option B — model flag (no suite required)

```bash
python run.py --runner nvidia_vllm_bc2ddb31 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --serve
```

### Full example with all options

```bash
python run.py --runner nvidia_vllm_bc2ddb31 \
    --suite suite_A \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --model-path /path/to/local/weights \
    --max-tokens 4096 \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    --serve \
    --port 8000 \
    --host 0.0.0.0 \
    --workers 4 \
    --api-key sk-mykey
```

When `--suite` and `--model` are both given, `--model` overrides the suite's model ID.
Explicit `--max-tokens` / `--max-model-len` always override suite values.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--serve` | — | Start in serve mode instead of benchmark mode |
| `--suite` | none | Suite ID (e.g. `suite_A`) — defines model and generation params. Optional if `--model` is given. |
| `--model` | none | HuggingFace model ID or name. Required if `--suite` is not given; overrides suite model ID if both are given. |
| `--model-path` | auto | Local path to model weights. Overrides HF download; falls back to `configs/models_local.yaml`. |
| `--max-tokens` | `2048` | Max output tokens per request. Overrides suite value if given. |
| `--max-model-len` | framework default | Max model context length. Leave unset to let vLLM read it from the model config. |
| `--tensor-parallel-size` | `1` | Number of GPUs for tensor parallelism |
| `--port` | `8000` | HTTP port |
| `--host` | `0.0.0.0` | Bind address |
| `--workers` | `4` | Max concurrent in-flight requests |
| `--api-key` | none | Require `Authorization: Bearer <key>` on all requests |

---

## Startup output

```
Runner:  VLLMRunner (nvidia_vllm_bc2ddb31)
Suite:   suite_A
Model:   meta-llama/Meta-Llama-3-8B-Instruct
Path:    /models/Meta-Llama-3-8B-Instruct
Params:  max_tokens=2048

2026-03-22 10:41:03 | INFO  | ============================================================
2026-03-22 10:41:03 | INFO  | AccelMark Serve
2026-03-22 10:41:03 | INFO  |   Runner    : nvidia_vllm_bc2ddb31
2026-03-22 10:41:03 | INFO  |   Framework : vLLM 0.6.6
2026-03-22 10:41:03 | INFO  |   Model     : meta-llama/Meta-Llama-3-8B-Instruct
2026-03-22 10:41:03 | INFO  |   Endpoint  : http://0.0.0.0:8000
2026-03-22 10:41:03 | INFO  |   Workers   : 4 concurrent requests
2026-03-22 10:41:03 | INFO  |   Auth      : disabled (no --api-key set)
2026-03-22 10:41:03 | INFO  | Capacity estimate from suite_A result (2026-03-22, NVIDIA A100-SXM4-80GB):
2026-03-22 10:41:03 | INFO  |   Offline throughput : 5,321 tokens/sec
2026-03-22 10:41:03 | INFO  |   Online max QPS     : 5.0 (within 500ms TTFT SLA)
2026-03-22 10:41:03 | INFO  |   Interactive TTFT   : 68ms p99
2026-03-22 10:41:03 | INFO  | ============================================================
```

Capacity estimates come from prior benchmark results for this runner ID.
Run a benchmark suite first to populate them:

```bash
python run.py --runner nvidia_vllm_bc2ddb31 --suite suite_A --scenario all
```

---

## Endpoints

### `GET /health`

Always accessible (no auth required).

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "implementation_id": "nvidia_vllm_bc2ddb31",
  "uptime_seconds": 42
}
```

---

### `GET /v1/models`

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Meta-Llama-3-8B-Instruct",
      "object": "model",
      "created": 1742636463,
      "owned_by": "nvidia_vllm_bc2ddb31"
    }
  ]
}
```

---

### `POST /v1/chat/completions`

Standard OpenAI chat completions format.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 64
  }'
```

**Streaming:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Tell me a joke."}],
    "stream": true
  }'
```

---

### `POST /v1/completions` (legacy)

For clients that use the older completion format.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "prompt": "The capital of France is",
    "max_tokens": 16
  }'
```

---

## Using with OpenAI clients

Any OpenAI-compatible client works by pointing it at your local server:

**Python (openai SDK):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-mykey",  # or "none" if auth is disabled
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

**LangChain:**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
)
```

---

## Concurrency and `--workers`

`--workers N` limits how many requests run concurrently inside the server.
Requests beyond the limit are queued (not rejected) until a slot opens.

This is **not** multi-process — one model instance serves all requests.
The inference framework (e.g. vLLM) handles internal batching.

Recommended values:
- `1` — single-request, lowest latency, useful for debugging
- `4` — good default for interactive workloads
- Higher — match your measured `max_valid_qps` from Suite A results

---

## Authentication

With `--api-key sk-mykey`, all endpoints except `/health` require:

```
Authorization: Bearer sk-mykey
```

Without `--api-key`, the server accepts all requests with no auth.

---

## Notes and limitations

**Streaming format:** The server uses SSE (server-sent events) format
compatible with the OpenAI spec. Responses are currently sent as a single
chunk rather than token-by-token. True token streaming requires runner-level
support — see the roadmap in `docs/DEVELOPMENT.md`.

**Runners without streaming:** Runners that set `SUPPORTS_STREAMING = False`
cannot be used with `--serve`. The server will refuse to start and print
a clear error message.

**Model field:** Use the exact model ID shown in `/v1/models` as the `model`
field in your requests. Requests with a different model name return a 400 error.

---

## Running tests

```bash
# From repo root
pip install pytest
pytest serve/tests/ -v
```

No GPU required — tests use a mock runner.
