from __future__ import annotations

import pytest

from autotree_sdk import (
    BranchPrunedEvent,
    BranchStartedEvent,
    DoneEvent,
    SSEParseError,
    TraceInvariantError,
    TreeHTTPError,
    TreeParameters,
    TreeStreamError,
)


TREE = TreeParameters(policy="beam", branches=2, budget_tokens=32, scorer=None)


def test_chat_completions_passes_tree_extension(tree_client, mock_app) -> None:
    response = tree_client.completions(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        tree=TREE,
        temperature=0.2,
    )

    assert response.choices[0].message.content == "ok"
    assert response.tree is not None
    assert mock_app.requests[-1] == {
        "path": "/v1/chat/completions",
        "body": {
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
            "stream": False,
            "model": "test-model",
            "tree": {
                "policy": "beam",
                "branches": 2,
                "budget_tokens": 32,
                "scorer": None,
            },
        },
    }


def test_tree_completions_returns_typed_response(tree_client) -> None:
    response = tree_client.tree_completions(
        messages=[{"role": "user", "content": "question"}], tree=TREE
    )

    assert response.choices[0].message.content == "answer"
    assert response.tree.branch_count == 1
    assert response.tree.final_scores == {"root": 0.9}


def test_chat_completions_stream_is_sse_passthrough(tree_client, mock_app) -> None:
    chunks = list(
        tree_client.completions(
            messages=[{"role": "user", "content": "hello"}],
            tree=TREE,
            stream=True,
        )
    )

    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"]["content"] == "ok"
    assert chunks[1]["tree"]["branch_count"] == 1
    assert mock_app.requests[-1]["body"]["stream"] is True


def test_stream_tree_completions_handles_partial_chunks_keepalives_and_done(
    tree_client,
) -> None:
    events = list(
        tree_client.stream_tree_completions(
            messages=[{"role": "user", "content": "question"}], tree=TREE
        )
    )

    assert isinstance(events[0], BranchStartedEvent)
    assert any(isinstance(event, BranchPrunedEvent) for event in events)
    assert isinstance(events[-1], DoneEvent)
    assert len(events) == 7


def test_stream_missing_done_raises_typed_terminal_error(tree_client) -> None:
    with pytest.raises(TraceInvariantError, match="missing_done"):
        list(
            tree_client.stream_tree_completions(
                messages=[{"role": "user", "content": "question"}],
                tree=TREE,
                scenario="missing_done",
            )
        )


def test_stream_capacity_error_exposes_typed_backoff(tree_client) -> None:
    with pytest.raises(TreeStreamError) as exc_info:
        list(
            tree_client.stream_tree_completions(
                messages=[{"role": "user", "content": "question"}],
                tree=TREE,
                scenario="capacity_error",
            )
        )

    assert exc_info.value.code == "kv_capacity_exhausted"
    assert exc_info.value.param == "kv_pages"
    assert exc_info.value.retry_after_seconds == 2


def test_stream_usage_mismatch_raises_typed_error(tree_client) -> None:
    with pytest.raises(TraceInvariantError, match="usage_token_mismatch"):
        list(
            tree_client.stream_tree_completions(
                messages=[{"role": "user", "content": "question"}],
                tree=TREE,
                scenario="usage_mismatch",
            )
        )


def test_token_for_unknown_branch_raises_at_parse_time(tree_client) -> None:
    with pytest.raises(TraceInvariantError, match="unknown_branch"):
        list(
            tree_client.stream_tree_completions(
                messages=[{"role": "user", "content": "question"}],
                tree=TREE,
                scenario="unknown_branch",
            )
        )


@pytest.mark.parametrize(
    ("scenario", "violation"),
    [
        ("invalid_json", "invalid_sse_json"),
        ("invalid_event", "invalid_tree_event"),
    ],
)
def test_malformed_stream_raises_typed_parse_error(
    tree_client, scenario, violation
) -> None:
    with pytest.raises(SSEParseError, match=violation) as exc_info:
        list(
            tree_client.stream_tree_completions(
                messages=[{"role": "user", "content": "question"}],
                tree=TREE,
                scenario=scenario,
            )
        )

    assert exc_info.value.violation == violation


def test_failed_post_is_not_retried(tree_client, mock_app) -> None:
    with pytest.raises(TreeHTTPError, match="HTTP 503"):
        tree_client.tree_completions(
            messages=[{"role": "user", "content": "question"}],
            tree=TREE,
            scenario="http_error",
        )

    assert len(mock_app.requests) == 1
