"""
Integration tests for serve/server.py.

Uses FastAPI's TestClient to exercise the full request/response cycle
with a MockRunner. No GPU, no real model, no network required.
"""

import json
import pytest
from fastapi.testclient import TestClient

from serve.server import app, _state
from serve.tests.mock_runner import MockRunner, NoStreamingMockRunner


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_state():
    """Reset server state before each test."""
    import asyncio
    _state.runner            = None
    _state.model_id          = None
    _state.implementation_id = None
    _state.semaphore         = asyncio.Semaphore(4)
    _state.api_key           = None
    _state.created_at        = 0
    yield


@pytest.fixture
def runner():
    return MockRunner(response_text="Test response.", output_tokens=3, input_tokens=5)


@pytest.fixture
def client(runner):
    """TestClient with a loaded mock runner."""
    import asyncio
    import time
    _state.runner            = runner
    _state.model_id          = "meta-llama/Meta-Llama-3-8B-Instruct"
    _state.implementation_id = runner._compute_implementation_id()
    _state.semaphore         = asyncio.Semaphore(4)
    _state.api_key           = None
    _state.created_at        = int(time.time())
    return TestClient(app)


@pytest.fixture
def authed_client(runner):
    """TestClient with API key auth enabled."""
    import asyncio
    import time
    _state.runner            = runner
    _state.model_id          = "meta-llama/Meta-Llama-3-8B-Instruct"
    _state.implementation_id = runner._compute_implementation_id()
    _state.semaphore         = asyncio.Semaphore(4)
    _state.api_key           = "sk-test-key"
    _state.created_at        = int(time.time())
    return TestClient(app)


MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# ── /health ───────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == MODEL
    assert "implementation_id" in data
    assert "uptime_seconds" in data


# ── /v1/models ────────────────────────────────────────────────────────────────

def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == MODEL
    assert data["data"][0]["object"] == "model"


# ── /v1/chat/completions — non-streaming ─────────────────────────────────────

def test_chat_completion_non_streaming(client):
    resp = client.post("/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == MODEL
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] == 5
    assert data["usage"]["completion_tokens"] == 3
    assert data["usage"]["total_tokens"] == 8


def test_chat_completion_wrong_model(client):
    resp = client.post("/v1/chat/completions", json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 400
    assert "gpt-4" in resp.json()["detail"]


def test_chat_completion_id_prefix(client):
    resp = client.post("/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.json()["id"].startswith("chatcmpl-")


# ── /v1/chat/completions — streaming ─────────────────────────────────────────

def test_chat_completion_streaming(client):
    resp = client.post("/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }, headers={"Accept": "text/event-stream"})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = [l for l in resp.text.split("\n") if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"

    # First chunk should have role
    first = json.loads(lines[0][6:])
    assert first["choices"][0]["delta"].get("role") == "assistant"

    # Last real chunk (before DONE) should have finish_reason
    last_chunk = json.loads(lines[-2][6:])
    assert last_chunk["choices"][0]["finish_reason"] == "stop"


def test_streaming_chunks_are_valid_json(client):
    resp = client.post("/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    data_lines = [l[6:] for l in resp.text.split("\n")
                  if l.startswith("data: ") and l != "data: [DONE]"]
    for line in data_lines:
        parsed = json.loads(line)
        assert "choices" in parsed
        assert "id" in parsed


# ── /v1/completions (legacy) ──────────────────────────────────────────────────

def test_legacy_completions_string_prompt(client):
    resp = client.post("/v1/completions", json={
        "model": MODEL,
        "prompt": "Once upon a time",
        "stream": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "text_completion"
    assert data["choices"][0]["text"] == "Test response."
    assert data["id"].startswith("cmpl-")


def test_legacy_completions_list_prompt(client):
    resp = client.post("/v1/completions", json={
        "model": MODEL,
        "prompt": ["First prompt", "Second prompt"],
        "stream": False,
    })
    assert resp.status_code == 200
    # Takes first prompt
    assert resp.json()["object"] == "text_completion"


def test_legacy_completions_streaming(client):
    resp = client.post("/v1/completions", json={
        "model": MODEL,
        "prompt": "Hello",
        "stream": True,
    })
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "data: [DONE]" in resp.text


def test_legacy_completions_wrong_model(client):
    resp = client.post("/v1/completions", json={
        "model": "wrong-model",
        "prompt": "hi",
    })
    assert resp.status_code == 400


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_auth_required_missing_header(authed_client):
    resp = authed_client.get("/v1/models")
    assert resp.status_code == 401


def test_auth_required_wrong_key(authed_client):
    resp = authed_client.get("/v1/models",
                             headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401


def test_auth_correct_key(authed_client):
    resp = authed_client.get("/v1/models",
                             headers={"Authorization": "Bearer sk-test-key"})
    assert resp.status_code == 200


def test_health_no_auth_required(authed_client):
    """Health endpoint is always accessible regardless of auth config."""
    resp = authed_client.get("/health")
    assert resp.status_code == 200


def test_no_auth_when_not_configured(client):
    """With no api_key set, all endpoints work without Authorization header."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200


# ── start_server validation ───────────────────────────────────────────────────

def test_start_server_refuses_no_streaming_runner():
    """start_server should raise RuntimeError for runners without streaming."""
    from serve.server import start_server
    runner = NoStreamingMockRunner()
    with pytest.raises(RuntimeError, match="SUPPORTS_STREAMING"):
        start_server(runner, "some-model", port=9999)


# ── True token streaming ──────────────────────────────────────────────────────

@pytest.fixture
def token_streaming_client():
    """TestClient backed by a TokenStreamingMockRunner."""
    import asyncio
    import time
    from serve.tests.mock_runner import TokenStreamingMockRunner
    runner = TokenStreamingMockRunner(
        response_text="Hello from token stream.",
        output_tokens=5,
        input_tokens=4,
    )
    _state.runner            = runner
    _state.model_id          = MODEL
    _state.implementation_id = runner._compute_implementation_id()
    _state.semaphore         = asyncio.Semaphore(4)
    _state.api_key           = None
    _state.created_at        = int(time.time())
    return TestClient(app)


def test_token_stream_produces_multiple_chunks(token_streaming_client):
    """With inference_fn_token_stream, each word should be its own SSE chunk."""
    resp = token_streaming_client.post("/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert resp.status_code == 200

    data_lines = [l for l in resp.text.split("\n") if l.startswith("data: ")]
    # Remove [DONE]
    chunk_lines = [l for l in data_lines if l != "data: [DONE]"]

    # First chunk is the role chunk, last is the stop chunk.
    # Everything in between is content — should be more than one for a
    # multi-word response ("Hello from token stream." = 4 words)
    content_chunks = []
    for line in chunk_lines:
        payload = json.loads(line[6:])
        delta = payload["choices"][0]["delta"]
        if delta.get("content"):
            content_chunks.append(delta["content"])

    assert len(content_chunks) > 1, (
        f"Expected multiple content chunks for token streaming, got {len(content_chunks)}"
    )


def test_token_stream_reassembles_correctly(token_streaming_client):
    """Concatenating all content deltas should reconstruct the full response."""
    resp = token_streaming_client.post("/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })

    data_lines = [l for l in resp.text.split("\n") if l.startswith("data: ")]
    content_parts = []
    for line in data_lines:
        if line == "data: [DONE]":
            continue
        payload = json.loads(line[6:])
        delta = payload["choices"][0]["delta"]
        if delta.get("content"):
            content_parts.append(delta["content"])

    full_text = "".join(content_parts)
    assert full_text == "Hello from token stream."


def test_token_stream_ends_with_stop_and_done(token_streaming_client):
    """Last real chunk must have finish_reason=stop, followed by [DONE]."""
    resp = token_streaming_client.post("/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    lines = resp.text.strip().split("\n")
    assert lines[-1] == "data: [DONE]"

    # Second to last non-empty line is the stop chunk
    data_lines = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
    last_chunk = json.loads(data_lines[-1][6:])
    assert last_chunk["choices"][0]["finish_reason"] == "stop"


def test_fallback_when_no_token_stream(client):
    """MockRunner (no inference_fn_token_stream) should still produce valid SSE."""
    resp = client.post("/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert resp.status_code == 200
    assert "data: [DONE]" in resp.text

    # Should be one content chunk (single-chunk fallback)
    data_lines = [l for l in resp.text.split("\n") if l.startswith("data: ")]
    content_chunks = [
        l for l in data_lines
        if l != "data: [DONE]"
        and json.loads(l[6:])["choices"][0]["delta"].get("content")
    ]
    assert len(content_chunks) == 1


def test_legacy_completions_token_stream(token_streaming_client):
    """Legacy /v1/completions also uses token streaming when available."""
    resp = token_streaming_client.post("/v1/completions", json={
        "model": MODEL,
        "prompt": "Hello",
        "stream": True,
    })
    assert resp.status_code == 200

    data_lines = [l for l in resp.text.split("\n") if l.startswith("data: ")]
    content_chunks = [
        l for l in data_lines
        if l != "data: [DONE]"
        and json.loads(l[6:])["choices"][0]["delta"].get("content")
    ]
    # Multi-word response → multiple chunks
    assert len(content_chunks) > 1