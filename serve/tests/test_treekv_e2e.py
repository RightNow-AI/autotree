from __future__ import annotations

from collections.abc import AsyncIterator
import math

import httpx
import pytest
from openai import AsyncOpenAI, AsyncStream, BaseModel
from pydantic import ConfigDict

from autotree_serve import create_app


class TreeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class TreeEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str


@pytest.fixture(scope="module")
def treekv_app():
    pytest.importorskip("autotree_scheduler")
    from autotree_core.engine import TreeKVEngine

    return create_app(TreeKVEngine(model_id="gpt2"))


@pytest.fixture
async def treekv_openai_client(treekv_app) -> AsyncIterator[AsyncOpenAI]:
    transport = httpx.ASGITransport(app=treekv_app)
    async with httpx.AsyncClient(transport=transport) as raw_client:
        yield AsyncOpenAI(
            api_key="test-key",
            base_url="http://test/v1",
            http_client=raw_client,
        )


def payload(*, stream: bool) -> dict[str, object]:
    return {
        "model": "gpt2",
        "messages": [{"role": "user", "content": "Name one color."}],
        "max_tokens": 4,
        "seed": 2026,
        "temperature": 0.0,
        "stream": stream,
        "tree": {
            "policy": "beam",
            "branches": 2,
            "budget_tokens": 6,
            "scorer": None,
        },
    }


async def test_official_openai_client_tree_completion_non_stream(treekv_openai_client) -> None:
    completion = await treekv_openai_client.post(
        "/tree/completions",
        body=payload(stream=False),
        cast_to=TreeResponse,
    )
    body = completion.model_dump()

    assert body["choices"][0]["message"]["content"]
    tree = body["tree"]
    assert tree["branch_count"] > 1
    assert tree["winner_branch_id"] in tree["final_scores"]
    assert set(tree["final_scores"]) == set(tree["tokens_spent_per_branch"])
    assert tree["kv_reuse_ratio"] > 1.0
    assert body["usage"]["completion_tokens"] == sum(
        tree["tokens_spent_per_branch"].values()
    )


async def test_official_openai_client_tree_completion_stream_usage(treekv_openai_client) -> None:
    stream = await treekv_openai_client.post(
        "/tree/completions",
        body=payload(stream=True),
        cast_to=TreeEvent,
        stream=True,
        stream_cls=AsyncStream[TreeEvent],
    )
    events = [event async for event in stream]

    assert events[-1].type == "done"
    raw_events = [event.model_dump() for event in events]
    tokens = [event for event in raw_events if event["type"] == "token"]
    assert all(math.isfinite(event["logprob"]) for event in tokens)
    assert all(
        event["reason"]
        for event in raw_events
        if event["type"] == "branch_pruned"
    )
    done = raw_events[-1]
    assert done["usage"]["completion_tokens"] == len(tokens)
    assert sum(done["tree"]["tokens_spent_per_branch"].values()) == len(tokens)
    assert done["tree"]["kv_reuse_ratio"] > 1.0


async def test_official_openai_client_same_seed_is_deterministic(
    treekv_openai_client,
) -> None:
    first = await treekv_openai_client.post(
        "/tree/completions",
        body=payload(stream=False),
        cast_to=TreeResponse,
    )
    second = await treekv_openai_client.post(
        "/tree/completions",
        body=payload(stream=False),
        cast_to=TreeResponse,
    )

    first_body = first.model_dump()
    second_body = second.model_dump()
    assert first_body["choices"][0]["message"]["content"] == second_body["choices"][0][
        "message"
    ]["content"]
    assert first_body["tree"]["winner_branch_id"] == second_body["tree"][
        "winner_branch_id"
    ]
    assert first_body["tree"]["final_scores"] == second_body["tree"]["final_scores"]
