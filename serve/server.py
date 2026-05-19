"""
AccelMark Serve — OpenAI-compatible inference server.

Wraps any RunnerProtocol-compatible runner as an OpenAI-compatible HTTP API.
Supports /v1/chat/completions, /v1/completions (legacy), and /v1/models.

Entry point (called by run.py):
    from serve.server import start_server
    start_server(runner, model_id, args)

Endpoints:
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions   (streaming + non-streaming)
    POST /v1/completions        (legacy, wraps chat completions internally)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from serve.adapter import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ModelCard,
    ModelList,
    make_chat_response,
    make_completion_response,
    make_sse_chunk,
    messages_to_prompt,
    sse_done,
)
from serve.capacity import format_capacity_log, load_capacity_estimate
from runners.protocol import RunnerProtocol
from runners.benchmark_runner import InferenceRequest

logger = logging.getLogger("accelmark.serve")


# ── Server state (set at startup by start_server) ─────────────────────────────

class _ServerState:
    runner: RunnerProtocol
    model_id: str
    implementation_id: Optional[str]
    semaphore: asyncio.Semaphore
    api_key: Optional[str]
    created_at: int


_state = _ServerState()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup already done by start_server before uvicorn launches
    logger.info("Server ready.")
    yield
    # Shutdown
    logger.info("Shutting down — releasing model resources...")
    try:
        _state.runner.release_resources()
        logger.info("Resources released.")
    except Exception as e:
        logger.warning(f"Error releasing resources: {e}")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AccelMark Serve",
    description="OpenAI-compatible inference API backed by an AccelMark runner.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Auth ──────────────────────────────────────────────────────────────────────

async def verify_api_key(authorization: Optional[str] = Header(default=None)):
    """Optional API key check. Skipped if no key was configured at startup."""
    if not _state.api_key:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Expected: Bearer <api-key>",
        )
    token = authorization.removeprefix("Bearer ").strip()
    if token != _state.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )


# ── Logging middleware ────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t_start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t_start) * 1000)
    logger.info(
        f"{request.method} {request.url.path} | {response.status_code} | {elapsed_ms}ms"
    )
    return response


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": _state.model_id,
        "implementation_id": _state.implementation_id,
        "uptime_seconds": int(time.time()) - _state.created_at,
    }


# ── Models ────────────────────────────────────────────────────────────────────

@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models() -> ModelList:
    return ModelList(
        data=[
            ModelCard(
                id=_state.model_id,
                created=_state.created_at,
                owned_by=_state.implementation_id or "accelmark",
            )
        ]
    )


# ── Chat completions ──────────────────────────────────────────────────────────

@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)], response_model=None)
async def chat_completions(
    req: ChatCompletionRequest,
) -> Union[ChatCompletionResponse, StreamingResponse]:

    if req.model != _state.model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{req.model}' not loaded. Loaded: '{_state.model_id}'",
        )

    # Build prompt — runner's format_prompt handles chat templates
    raw_prompt = messages_to_prompt(req.messages)
    prompt     = _state.runner.format_prompt(raw_prompt)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async def _run_inference():
        """Acquire semaphore and call runner."""
        async with _state.semaphore:
            t0 = time.perf_counter()
            result = await _state.runner.inference_fn_streaming(
                InferenceRequest(prompt=prompt)
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000)

        content           = result.output_text or ""
        prompt_tokens     = result.input_tokens or 0
        completion_tokens = result.output_tokens or 0

        logger.info(
            f"POST /v1/chat/completions | "
            f"{prompt_tokens} in → {completion_tokens} out | "
            f"{elapsed_ms}ms"
            + (f" | TTFT {result.first_token_time_ms:.0f}ms"
               if result.first_token_time_ms else "")
        )
        return content, prompt_tokens, completion_tokens

    if req.stream:
        async def _stream_generator():
            # Role chunk first
            yield make_sse_chunk(completion_id, _state.model_id,
                                 content=None, role="assistant")

            # Try true token streaming — falls back to single-chunk if not supported
            try:
                async with _state.semaphore:
                    t0 = time.perf_counter()
                    token_count   = 0
                    first_yielded = False
                    async for token in _state.runner.inference_fn_token_stream(
                        InferenceRequest(prompt=prompt)
                    ):
                        if not first_yielded:
                            ttft_ms = round((time.perf_counter() - t0) * 1000)
                            logger.info(
                                f"POST /v1/chat/completions [stream] | "
                                f"TTFT {ttft_ms}ms"
                            )
                            first_yielded = True
                        token_count += 1
                        yield make_sse_chunk(completion_id, _state.model_id,
                                             content=token)
                    total_ms = round((time.perf_counter() - t0) * 1000)
                    logger.info(
                        f"POST /v1/chat/completions [stream] | "
                        f"{token_count} tokens | {total_ms}ms total"
                    )
            except NotImplementedError:
                # Runner doesn't support token streaming — send full response
                # as one content chunk.
                content, prompt_tokens, completion_tokens = await _run_inference()
                if content:
                    yield make_sse_chunk(completion_id, _state.model_id,
                                         content=content)

            # Stop chunk
            yield make_sse_chunk(completion_id, _state.model_id,
                                 content=None, finish_reason="stop")
            yield sse_done()

        return StreamingResponse(
            _stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming
    content, prompt_tokens, completion_tokens = await _run_inference()
    return make_chat_response(_state.model_id, content, prompt_tokens, completion_tokens)


# ── Legacy completions ────────────────────────────────────────────────────────

@app.post("/v1/completions", dependencies=[Depends(verify_api_key)], response_model=None)
async def completions(
    req: CompletionRequest,
) -> Union[CompletionResponse, StreamingResponse]:
    """
    Legacy /v1/completions endpoint. Wraps chat completions internally
    by treating the prompt as a single user message.
    """
    if req.model != _state.model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{req.model}' not loaded. Loaded: '{_state.model_id}'",
        )

    # Normalise prompt (can be str or list[str] per spec — we take the first)
    raw_prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    prompt     = _state.runner.format_prompt(raw_prompt)

    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"

    async def _run():
        async with _state.semaphore:
            t0 = time.perf_counter()
            result = await _state.runner.inference_fn_streaming(
                InferenceRequest(prompt=prompt)
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000)

        content           = result.output_text or ""
        prompt_tokens     = result.input_tokens or 0
        completion_tokens = result.output_tokens or 0

        logger.info(
            f"POST /v1/completions | "
            f"{prompt_tokens} in → {completion_tokens} out | {elapsed_ms}ms"
        )
        return content, prompt_tokens, completion_tokens

    if req.stream:
        async def _stream():
            yield make_sse_chunk(completion_id, _state.model_id,
                                 content=None, role="assistant")

            try:
                async with _state.semaphore:
                    async for token in _state.runner.inference_fn_token_stream(
                        InferenceRequest(prompt=prompt)
                    ):
                        yield make_sse_chunk(completion_id, _state.model_id,
                                             content=token)
            except NotImplementedError:
                # Single-chunk fallback
                content, _, _ = await _run()
                if content:
                    yield make_sse_chunk(completion_id, _state.model_id,
                                         content=content)

            yield make_sse_chunk(completion_id, _state.model_id,
                                 content=None, finish_reason="stop")
            yield sse_done()

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    content, prompt_tokens, completion_tokens = await _run()
    return make_completion_response(
        _state.model_id, content, prompt_tokens, completion_tokens
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def start_server(
    runner: RunnerProtocol,
    model_id: str,
    port: int = 8000,
    host: str = "0.0.0.0",
    workers: int = 4,
    api_key: Optional[str] = None,
) -> None:
    """
    Load model, log startup info, then launch uvicorn.
    Called by run.py after the runner is instantiated.

    Args:
        runner:   Any RunnerProtocol-compatible runner (already instantiated,
                  load_model() already called by run.py before start_server)
        model_id: HuggingFace model ID or local path (used as the OpenAI
                  'model' field in all responses)
        port:     HTTP port to listen on
        host:     Bind address (default 0.0.0.0 = all interfaces)
        workers:  Max concurrent in-flight requests (semaphore size)
        api_key:  If set, all endpoints require Authorization: Bearer <key>
    """
    # ── Validate runner capability ─────────────────────────────────────────
    if not runner.SUPPORTS_STREAMING:
        raise RuntimeError(
            f"Runner '{runner.__class__.__name__}' sets SUPPORTS_STREAMING = False.\n"
            f"The serve API requires streaming support.\n"
            f"Serving is not available for this runner."
        )

    # ── Configure logging ──────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Populate server state ──────────────────────────────────────────────
    _state.runner            = runner
    _state.model_id          = model_id
    _state.implementation_id = runner._compute_implementation_id()
    _state.semaphore         = asyncio.Semaphore(workers)
    _state.api_key           = api_key
    _state.created_at        = int(time.time())

    # ── Startup log ────────────────────────────────────────────────────────
    impl_id = _state.implementation_id or "unknown"
    logger.info("=" * 60)
    logger.info("AccelMark Serve")
    logger.info(f"  Runner    : {impl_id}")
    logger.info(f"  Framework : {runner._get_framework_name()} "
                f"{runner._get_framework_version()}")
    logger.info(f"  Model     : {model_id}")
    logger.info(f"  Endpoint  : http://{host}:{port}")
    logger.info(f"  Workers   : {workers} concurrent requests")
    logger.info(f"  Auth      : {'enabled' if api_key else 'disabled (no --api-key set)'}")

    # ── Capacity estimate ──────────────────────────────────────────────────
    if impl_id != "unknown":
        try:
            est = load_capacity_estimate(impl_id)
            if est:
                for line in format_capacity_log(est):
                    logger.info(line)
            else:
                logger.info(
                    "No prior benchmark results found for this runner. "
                    "Run a benchmark suite first to get capacity estimates."
                )
        except Exception as e:
            logger.warning(f"Could not load capacity estimate: {e}")

    logger.info("=" * 60)

    # ── Launch uvicorn ─────────────────────────────────────────────────────
    # Imported lazily so importing `serve.server` (e.g. from tests, or to
    # build the ASGI `app` for an external runner) does not require uvicorn.
    import uvicorn

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning",   # uvicorn's own logs are noisy; we handle ours
        access_log=False,      # we log via middleware instead
    )