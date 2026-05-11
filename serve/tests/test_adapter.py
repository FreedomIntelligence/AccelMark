"""
Unit tests for serve/adapter.py.

Tests prompt construction, response model serialisation, and SSE formatting.
No server or GPU required.
"""

import json
import pytest

from serve.adapter import (
    ChatMessage,
    ChatCompletionRequest,
    CompletionRequest,
    make_chat_response,
    make_completion_response,
    make_sse_chunk,
    messages_to_prompt,
    sse_done,
)


# ── messages_to_prompt ────────────────────────────────────────────────────────

def test_messages_to_prompt_single_user():
    msgs = [ChatMessage(role="user", content="Hello")]
    prompt = messages_to_prompt(msgs)
    assert "<|im_start|>user" in prompt
    assert "Hello" in prompt
    assert prompt.endswith("<|im_start|>assistant")


def test_messages_to_prompt_system_and_user():
    msgs = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Tell me a joke."),
    ]
    prompt = messages_to_prompt(msgs)
    assert "<|im_start|>system" in prompt
    assert "You are helpful." in prompt
    assert "<|im_start|>user" in prompt
    assert "Tell me a joke." in prompt
    assert prompt.endswith("<|im_start|>assistant")


def test_messages_to_prompt_multi_turn():
    msgs = [
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello!"),
        ChatMessage(role="user", content="How are you?"),
    ]
    prompt = messages_to_prompt(msgs)
    # All three turns should appear
    assert prompt.count("<|im_start|>") == 4  # 3 messages + trailing assistant
    assert "How are you?" in prompt


# ── make_chat_response ────────────────────────────────────────────────────────

def test_make_chat_response_structure():
    resp = make_chat_response(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        content="Hello there!",
        prompt_tokens=10,
        completion_tokens=3,
    )
    assert resp.object == "chat.completion"
    assert resp.model == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert len(resp.choices) == 1
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content == "Hello there!"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 3
    assert resp.usage.total_tokens == 13


def test_make_chat_response_id_format():
    resp = make_chat_response("model", "text", 0, 0)
    assert resp.id.startswith("chatcmpl-")


def test_make_chat_response_serialisable():
    resp = make_chat_response("model", "text", 5, 3)
    # Should not raise
    data = json.loads(resp.model_dump_json())
    assert data["object"] == "chat.completion"


# ── make_completion_response ──────────────────────────────────────────────────

def test_make_completion_response_structure():
    resp = make_completion_response(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        text="The sky is blue.",
        prompt_tokens=5,
        completion_tokens=4,
    )
    assert resp.object == "text_completion"
    assert resp.choices[0].text == "The sky is blue."
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.total_tokens == 9


def test_make_completion_response_id_format():
    resp = make_completion_response("model", "text", 0, 0)
    assert resp.id.startswith("cmpl-")


# ── SSE helpers ───────────────────────────────────────────────────────────────

def test_make_sse_chunk_format():
    chunk = make_sse_chunk("chatcmpl-abc123", "my-model", content="Hello")
    assert chunk.startswith("data: ")
    assert chunk.endswith("\n\n")
    payload = json.loads(chunk[6:])
    assert payload["object"] == "chat.completion.chunk"
    assert payload["id"] == "chatcmpl-abc123"
    assert payload["model"] == "my-model"
    assert payload["choices"][0]["delta"]["content"] == "Hello"
    assert payload["choices"][0]["finish_reason"] is None


def test_make_sse_chunk_role_only():
    chunk = make_sse_chunk("id", "model", content=None, role="assistant")
    payload = json.loads(chunk[6:])
    assert payload["choices"][0]["delta"]["role"] == "assistant"
    assert payload["choices"][0]["delta"].get("content") is None


def test_make_sse_chunk_stop():
    chunk = make_sse_chunk("id", "model", content=None, finish_reason="stop")
    payload = json.loads(chunk[6:])
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_sse_done():
    assert sse_done() == "data: [DONE]\n\n"


# ── Request model validation ──────────────────────────────────────────────────

def test_chat_request_defaults():
    req = ChatCompletionRequest(
        model="llama3",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.max_tokens == 512
    assert req.stream is False
    assert req.temperature == 1.0


def test_completion_request_prompt_string():
    req = CompletionRequest(model="llama3", prompt="Hello world")
    assert req.prompt == "Hello world"


def test_completion_request_prompt_list():
    req = CompletionRequest(model="llama3", prompt=["Hello", "World"])
    assert req.prompt == ["Hello", "World"]
