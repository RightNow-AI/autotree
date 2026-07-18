from __future__ import annotations

import pytest

from autotree_serve.cli import main
from autotree_serve.engine import (
    DeterministicEngine,
    GenerationDone,
    GenerationRequest,
    Message,
    TreeExecution,
)


def test_cli_help_is_honest_about_deterministic_engine(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "seeded toy generator" in output
    assert "does not serve real model weights" in output


def test_treekv_engine_fails_loudly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--model", "demo", "--engine", "treekv"])

    assert exc.value.code != 0
    error = capsys.readouterr().err
    assert "Tree-KV engine integration lands in Phase 2" in error
    assert "scheduler" in error
    assert "GPU runtime" in error
    assert "real-model weight loader" in error


def test_cli_starts_deterministic_server(monkeypatch):
    called = {}

    def fake_run(app, *, host, port):
        called.update(app=app, host=host, port=port)

    monkeypatch.setattr("autotree_serve.cli.uvicorn.run", fake_run)
    main(
        [
            "serve",
            "--model",
            "toy-model",
            "--engine",
            "deterministic",
            "--host",
            "0.0.0.0",
            "--port",
            "8123",
        ]
    )

    assert called["host"] == "0.0.0.0"
    assert called["port"] == 8123
    assert called["app"].state.engine.model_metadata.id == "toy-model"
    assert called["app"].state.engine.model_metadata.real_model_weights is False


async def test_deterministic_engine_repeats_seeded_event_stream():
    engine = DeterministicEngine("toy")
    request = GenerationRequest(
        model="toy",
        messages=(Message(role="user", content="repeat this"),),
        max_tokens=5,
        temperature=1.0,
        seed=123,
        tree=TreeExecution(
            policy="beam",
            branches=3,
            budget_tokens=11,
            scorer=None,
        ),
    )

    first = [event async for event in engine.generate(request)]
    second = [event async for event in engine.generate(request)]
    first_done = next(event for event in first if isinstance(event, GenerationDone))
    second_done = next(event for event in second if isinstance(event, GenerationDone))

    assert first_done.text == second_done.text
    assert first_done.usage == second_done.usage
    assert first_done.tree_summary == second_done.tree_summary
    assert first_done.usage.completion_tokens == 11


async def test_tree_winner_has_generated_content_when_budget_is_narrow():
    engine = DeterministicEngine("toy")
    request = GenerationRequest(
        model="toy",
        messages=(Message(role="user", content="use the only generated token"),),
        max_tokens=16,
        temperature=1.0,
        seed=1,
        tree=TreeExecution(
            policy="beam",
            branches=4,
            budget_tokens=1,
            scorer=None,
        ),
    )

    events = [event async for event in engine.generate(request)]
    done = next(event for event in events if isinstance(event, GenerationDone))

    assert done.text
    assert done.tree_summary is not None
    assert done.tree_summary.tokens_spent_per_branch[done.branch_id] == 1
