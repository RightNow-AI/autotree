from __future__ import annotations

import pytest

from autotree_sdk import ExportError, TreeClient, rollout

from .mock_asgi import MockAutoTreeASGI, make_http_client


def test_rollout_end_to_end_and_rl_exports() -> None:
    app = MockAutoTreeASGI()
    http_client = make_http_client(app)
    client = TreeClient("http://autotree.test", http_client=http_client)
    try:
        batch = rollout(
            ["first", [{"role": "user", "content": "second"}]],
            2,
            policy="best_first",
            budget_tokens=64,
            seed=7,
            model="test-model",
            client=client,
        )
    finally:
        http_client.close()

    assert len(batch.trees) == 2
    root = batch.trees[0].branch("root")
    alt = batch.trees[0].branch("alt")
    assert root.completion == "answer"
    assert root.token_ids == [None, None]
    assert root.token_logprobs == [-0.1, -0.2]
    assert root.cumulative_logprob == -0.30000000000000004
    assert alt.branch_path == ["root", "alt"]
    assert alt.pruned is True

    grpo = batch.to_grpo_samples()
    assert len(grpo) == 2
    assert grpo[0]["prompt"] == "first"
    assert grpo[0]["completion"] == "answer"
    assert grpo[0]["branch_path"] == ["root"]
    assert grpo[0]["pruned"] is False

    diagnostic = batch.to_grpo_samples(include_pruned=True)
    assert len(diagnostic) == 4
    assert any(sample["pruned"] for sample in diagnostic)

    pairs = batch.to_rlhf_pairs()
    assert len(pairs) == 2
    assert pairs[0]["chosen"]["branch_id"] == "root"
    assert pairs[0]["rejected"]["branch_id"] == "alt"
    assert pairs[0]["chosen_score"] == 0.9
    assert app.requests[0]["body"]["tree"] == {
        "policy": "best_first",
        "branches": 2,
        "budget_tokens": 64,
        "scorer": None,
    }
    assert app.requests[0]["body"]["seed"] == 7


def test_rlhf_export_rejects_positional_final_scores() -> None:
    app = MockAutoTreeASGI()
    http_client = make_http_client(app)
    client = TreeClient("http://autotree.test", http_client=http_client)
    try:
        batch = rollout(
            ["prompt"],
            2,
            client=client,
            scenario="positional_scores",
        )
    finally:
        http_client.close()

    assert batch.trees[0].tree_summary.final_scores == [0.9, 0.1]
    with pytest.raises(ExportError, match="ambiguous_final_scores"):
        batch.to_rlhf_pairs()
