"""
AccelMark Serve — OpenAI format adapter

Translates between OpenAI API request/response format and the
RunnerProtocol interface. No knowledge of FastAPI or BenchmarkRunner here —
only Pydantic models and pure conversion functions.
"""

from __future__ import annotations

import time
import uuid
from typing import AsyncIterator, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── OpenAI request models ─────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[Union[str, list[str]]] = None
    n: Optional[int] = 1
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    """Legacy /v1/completions format."""
    model: str
    prompt: Union[str, list[str]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[Union[str, list[str]]] = None
    n: Optional[int] = 1
    user: Optional[str] = None


# ── OpenAI response models ────────────────────────────────────────────────────

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "error"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length", "error"]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Literal["stop", "length", "error"] = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "accelmark"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


# ── Prompt construction ───────────────────────────────────────────────────────

def messages_to_prompt(messages: list[ChatMessage]) -> str:
    """
    Convert a list of chat messages to a single prompt string.

    Uses a simple ChatML-style format. Runners that support native chat
    templates should override format_prompt() instead — this is just the
    fallback for runners that accept raw strings.

    Format:
        <|im_start|>system
        {system message}<|im_end|>
        <|im_start|>user
        {user message}<|im_end|>
        <|im_start|>assistant
    """
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant")
    return "\n".join(parts)


# ── Response construction ─────────────────────────────────────────────────────

def make_chat_response(
    model_id: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_id,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def make_completion_response(
    model_id: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> CompletionResponse:
    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_id,
        choices=[CompletionChoice(text=text, finish_reason="stop")],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def make_sse_chunk(
    completion_id: str,
    model_id: str,
    content: Optional[str],
    finish_reason: Optional[str] = None,
    role: Optional[str] = None,
) -> str:
    """Serialize one SSE data line for a streaming chat completion chunk."""
    chunk = ChatCompletionChunk(
        id=completion_id,
        created=int(time.time()),
        model=model_id,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(role=role, content=content),
                finish_reason=finish_reason,
            )
        ],
    )
    return f"data: {chunk.model_dump_json()}\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"
