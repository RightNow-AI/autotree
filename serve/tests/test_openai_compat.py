from __future__ import annotations

from conftest import MODEL_ID


async def test_official_openai_client_non_stream_needs_only_base_url(openai_client):
    completion = await openai_client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "explain tree reuse"}],
        max_tokens=6,
        seed=7,
    )

    assert completion.object == "chat.completion"
    assert completion.model == MODEL_ID
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].message.content
    assert completion.usage is not None
    assert completion.usage.completion_tokens == 6
    assert completion.usage.total_tokens == (
        completion.usage.prompt_tokens + completion.usage.completion_tokens
    )


async def test_official_openai_client_stream_chunks_and_usage(openai_client):
    stream = await openai_client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "stream a deterministic answer"}],
        max_tokens=5,
        seed=11,
        stream=True,
    )
    chunks = [chunk async for chunk in stream]

    assert chunks
    assert all(chunk.object == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0].choices[0].delta.role == "assistant"
    content = "".join(
        choice.delta.content or ""
        for chunk in chunks
        for choice in chunk.choices
    )
    assert content
    assert any(
        choice.finish_reason == "length"
        for chunk in chunks
        for choice in chunk.choices
    )
    usage_chunks = [chunk for chunk in chunks if chunk.usage is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].usage.completion_tokens == 5


async def test_tree_extra_body_is_accepted_and_reflected(openai_client):
    completion = await openai_client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "compare candidate paths"}],
        max_tokens=5,
        seed=3,
        extra_body={
            "tree": {
                "policy": "beam",
                "branches": 4,
                "budget_tokens": 13,
                "scorer": "toy-score",
            }
        },
    )

    tree = completion.model_extra["tree"]
    assert tree["policy"] == "beam"
    assert tree["branch_count"] == 4
    assert tree["scorer"] == "toy-score"
    assert sum(tree["tokens_spent_per_branch"].values()) == 13
    assert completion.usage.completion_tokens == 13


async def test_models_are_honest_about_deterministic_toy_engine(http_client):
    response = await http_client.get("/v1/models")

    assert response.status_code == 200
    model = response.json()["data"][0]
    assert model["id"] == MODEL_ID
    assert model["metadata"]["engine"] == "deterministic"
    assert model["metadata"]["real_model_weights"] is False
    assert "toy generator" in model["metadata"]["description"]
