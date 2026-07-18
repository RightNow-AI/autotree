"""FastAPI application factory and OpenAI-compatible wire adapters."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

from .engine import (
    BranchMerged,
    BranchPruned,
    BranchStarted,
    DeterministicEngine,
    EngineEvent,
    EngineProtocol,
    GenerationDone,
    GenerationRequest,
    Message,
    TokenGenerated,
    TreeExecution,
)
from .metrics import ServeMetrics
from .schema import ChatCompletionRequest, TreeCompletionRequest


class EngineContractError(RuntimeError):
    """Raised when an engine violates the public event/accounting contract."""


class EventAccumulator:
    def __init__(self) -> None:
        self.started: set[str] = set()
        self.terminal: set[str] = set()
        self.tokens: dict[str, list[str]] = {}
        self.token_count = 0
        self.finished = False

    def accept(self, event: EngineEvent) -> None:
        if self.finished:
            raise EngineContractError("engine emitted an event after done")
        if isinstance(event, BranchStarted):
            if event.branch_id in self.started:
                raise EngineContractError(f"branch {event.branch_id} started more than once")
            self.started.add(event.branch_id)
            self.tokens[event.branch_id] = []
            return
        if isinstance(event, TokenGenerated):
            if event.branch_id not in self.started:
                raise EngineContractError(f"token emitted for unknown branch {event.branch_id}")
            if event.branch_id in self.terminal:
                raise EngineContractError(f"token emitted after branch {event.branch_id} terminated")
            self.tokens[event.branch_id].append(event.token)
            self.token_count += 1
            return
        if isinstance(event, (BranchPruned, BranchMerged)):
            self._terminate(event.branch_id)
            return
        if isinstance(event, GenerationDone):
            self._terminate(event.branch_id)
            if self.started != self.terminal:
                missing = sorted(self.started - self.terminal)
                raise EngineContractError(f"branches without terminal events: {missing}")
            if event.usage.completion_tokens != self.token_count:
                raise EngineContractError(
                    "completion token usage does not match emitted token events: "
                    f"usage={event.usage.completion_tokens}, events={self.token_count}"
                )
            emitted_text = "".join(self.tokens[event.branch_id])
            if emitted_text != event.text:
                raise EngineContractError("winning text does not match winner token events")
            self.finished = True

    def _terminate(self, branch_id: str) -> None:
        if branch_id not in self.started:
            raise EngineContractError(f"terminal event for unknown branch {branch_id}")
        if branch_id in self.terminal:
            raise EngineContractError(f"branch {branch_id} terminated more than once")
        self.terminal.add(branch_id)


def create_app(
    engine: EngineProtocol | None = None,
    *,
    model_id: str = "autotree-deterministic",
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    selected_engine = engine or DeterministicEngine(model_id=model_id)
    metrics = ServeMetrics(registry)
    app = FastAPI(title="autotree-serve", version="0.1.0")
    app.state.engine = selected_engine
    app.state.metrics = metrics

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = first.get("loc", ())
        param = ".".join(str(part) for part in location if part != "body") or None
        return _openai_error(
            status_code=422,
            message=first.get("msg", "Request validation failed"),
            param=param,
            code="validation_error",
        )

    @app.exception_handler(EngineContractError)
    async def engine_contract_error_handler(
        _request: Request,
        exc: EngineContractError,
    ) -> JSONResponse:
        return _openai_error(
            status_code=500,
            message=f"Engine event contract violation: {exc}",
            error_type="server_error",
            code="engine_contract_error",
        )

    @app.middleware("http")
    async def count_requests(request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        endpoint = request.url.path
        metrics.requests_total.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        return response

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        metadata = selected_engine.model_metadata
        return {
            "object": "list",
            "data": [
                {
                    "id": metadata.id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "autotree",
                    "metadata": {
                        "engine": metadata.engine,
                        "description": metadata.description,
                        "real_model_weights": metadata.real_model_weights,
                        "tree_policies": list(metadata.tree_policies),
                    },
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest) -> Response:
        model_error = _validate_model(body.model, selected_engine)
        if model_error:
            return model_error
        request = _to_engine_request(body)
        if body.stream:
            return StreamingResponse(
                _chat_stream(selected_engine, request, metrics),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        _events, done = await _collect_events(selected_engine, request, metrics)
        return JSONResponse(_completion_response(done, body.model))

    @app.post("/v1/tree/completions")
    async def tree_completions(body: TreeCompletionRequest) -> Response:
        model_error = _validate_model(body.model, selected_engine)
        if model_error:
            return model_error
        request = _to_engine_request(body)
        if body.stream:
            return StreamingResponse(
                _tree_stream(selected_engine, request, metrics),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        _events, done = await _collect_events(selected_engine, request, metrics)
        return JSONResponse(_completion_response(done, body.model))

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(
            content=generate_latest(metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


def _validate_model(model: str, engine: EngineProtocol) -> JSONResponse | None:
    if model == engine.model_metadata.id:
        return None
    return _openai_error(
        status_code=404,
        message=f"Model '{model}' is not served by this process.",
        param="model",
        code="model_not_found",
    )


def _to_engine_request(body: ChatCompletionRequest) -> GenerationRequest:
    tree = None
    if body.tree is not None:
        tree = TreeExecution(
            policy=body.tree.policy,
            branches=body.tree.branches,
            budget_tokens=body.tree.budget_tokens,
            scorer=body.tree.scorer,
        )
    return GenerationRequest(
        model=body.model,
        messages=tuple(Message(role=item.role, content=item.content) for item in body.messages),
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        seed=body.seed,
        tree=tree,
    )


async def _collect_events(
    engine: EngineProtocol,
    request: GenerationRequest,
    metrics: ServeMetrics,
) -> tuple[list[EngineEvent], GenerationDone]:
    accumulator = EventAccumulator()
    events: list[EngineEvent] = []
    done: GenerationDone | None = None
    async for event in engine.generate(request):
        accumulator.accept(event)
        metrics.observe_event(event)
        events.append(event)
        if isinstance(event, GenerationDone):
            done = event
    if done is None:
        raise EngineContractError("engine stream ended without a done event")
    return events, done


def _completion_response(
    done: GenerationDone,
    model: str,
) -> dict[str, object]:
    response: dict[str, object] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": done.text},
                "logprobs": None,
                "finish_reason": done.finish_reason,
            }
        ],
        "usage": done.usage.to_dict(),
    }
    if done.tree_summary is not None:
        response["tree"] = done.tree_summary.to_dict()
    return response


async def _chat_stream(
    engine: EngineProtocol,
    request: GenerationRequest,
    metrics: ServeMetrics,
) -> AsyncIterator[str]:
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    yield _sse_data(
        _chat_chunk(
            stream_id,
            created,
            request.model,
            choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        )
    )

    if request.tree is not None:
        accumulator = EventAccumulator()
        done = None
        async for event in engine.generate(request):
            accumulator.accept(event)
            metrics.observe_event(event)
            if not isinstance(event, GenerationDone):
                yield _sse_data(
                    _chat_chunk(
                        stream_id,
                        created,
                        request.model,
                        choices=[],
                        tree_event=_event_payload(event),
                    )
                )
                continue
            done = event
        if done is None:
            raise EngineContractError("engine stream ended without a done event")
        if done.text:
            yield _sse_data(
                _chat_chunk(
                    stream_id,
                    created,
                    request.model,
                    choices=[{"index": 0, "delta": {"content": done.text}, "finish_reason": None}],
                )
            )
        yield _sse_data(
            _chat_chunk(
                stream_id,
                created,
                request.model,
                choices=[{"index": 0, "delta": {}, "finish_reason": done.finish_reason}],
                tree=done.tree_summary.to_dict() if done.tree_summary else None,
                tree_event=_event_payload(done),
            )
        )
    else:
        accumulator = EventAccumulator()
        done = None
        async for event in engine.generate(request):
            accumulator.accept(event)
            metrics.observe_event(event)
            if isinstance(event, TokenGenerated):
                yield _sse_data(
                    _chat_chunk(
                        stream_id,
                        created,
                        request.model,
                        choices=[
                            {
                                "index": 0,
                                "delta": {"content": event.token},
                                "finish_reason": None,
                            }
                        ],
                    )
                )
            elif isinstance(event, GenerationDone):
                done = event
                yield _sse_data(
                    _chat_chunk(
                        stream_id,
                        created,
                        request.model,
                        choices=[
                            {"index": 0, "delta": {}, "finish_reason": event.finish_reason}
                        ],
                    )
                )
        if done is None:
            raise EngineContractError("engine stream ended without a done event")

    yield _sse_data(
        _chat_chunk(
            stream_id,
            created,
            request.model,
            choices=[],
            usage=done.usage.to_dict(),
        )
    )
    yield "data: [DONE]\n\n"


async def _tree_stream(
    engine: EngineProtocol,
    request: GenerationRequest,
    metrics: ServeMetrics,
) -> AsyncIterator[str]:
    accumulator = EventAccumulator()
    saw_done = False
    async for event in engine.generate(request):
        accumulator.accept(event)
        metrics.observe_event(event)
        saw_done = saw_done or isinstance(event, GenerationDone)
        payload = _event_payload(event)
        yield f"event: {event.type}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
    if not saw_done:
        raise EngineContractError("engine stream ended without a done event")


def _chat_chunk(
    stream_id: str,
    created: int,
    model: str,
    *,
    choices: list[dict[str, object]],
    usage: dict[str, int] | None = None,
    tree: dict[str, object] | None = None,
    tree_event: dict[str, object] | None = None,
) -> dict[str, object]:
    chunk: dict[str, object] = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    if tree is not None:
        chunk["tree"] = tree
    if tree_event is not None:
        chunk["tree_event"] = tree_event
    return chunk


def _event_payload(event: EngineEvent) -> dict[str, object]:
    if isinstance(event, GenerationDone):
        payload: dict[str, object] = {
            "type": event.type,
            "branch_id": event.branch_id,
            "text": event.text,
            "finish_reason": event.finish_reason,
            "usage": event.usage.to_dict(),
            "counters": asdict(event.counters),
        }
        if event.tree_summary is not None:
            payload["tree"] = event.tree_summary.to_dict()
        return payload
    return {"type": event.type, **asdict(event)}


def _sse_data(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _openai_error(
    *,
    status_code: int,
    message: str,
    param: str | None = None,
    error_type: str = "invalid_request_error",
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )
