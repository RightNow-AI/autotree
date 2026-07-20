from __future__ import annotations

import asyncio
import math
from collections import deque
from dataclasses import asdict, replace

import pytest
import torch

from autotree_core.engine import (
    BranchMerged,
    BranchPruned,
    BranchStarted,
    GenerationDone,
    GenerationRequest,
    KVCapacityExceededError,
    Message,
    TokenGenerated,
    TreeExecution,
    TreeKVEngine,
)


class ScriptedScheduler:
    """Drive one fork, one killed sibling, and one winning continuation."""

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self._commands: deque[dict[str, object]] = deque()
        self._token_events = 0

    def feed_event(self, event: dict[str, object]) -> None:
        if event["type"] != "token_sampled":
            return
        self._token_events += 1
        if self._token_events == 1:
            self._commands.extend(
                [
                    {"type": "fork_at", "branch": 0, "width": 2},
                    {"type": "continue", "branch": 1},
                    {"type": "continue", "branch": 2},
                ]
            )
        elif self._token_events == 3:
            self._commands.extend(
                [
                    {"type": "kill", "branch": 2, "reason": "beam_pruned"},
                    {"type": "finalize", "branch": 1},
                    {"type": "kill", "branch": 0, "reason": "tree_budget_exhausted"},
                ]
            )

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands


class ExhaustionScheduler:
    def __init__(
        self,
        config: dict[str, object],
        observed_events: list[dict[str, object]],
    ) -> None:
        self.config = config
        self.observed_events = observed_events
        self._commands: deque[dict[str, object]] = deque()

    def feed_event(self, event: dict[str, object]) -> None:
        self.observed_events.append(event)
        if event["type"] == "token_sampled":
            self._commands.append({"type": "continue", "branch": event["branch"]})
        elif event["type"] == "branch_exhausted":
            self._commands.append({"type": "finalize", "branch": event["branch"]})

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands


class ContinueUntilCapacityScheduler:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self._commands: deque[dict[str, object]] = deque()

    def feed_event(self, event: dict[str, object]) -> None:
        if event["type"] == "token_sampled":
            self._commands.append({"type": "continue", "branch": event["branch"]})

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands


class ConvergingScheduler:
    """Fork two greedy-identical children, then finalize the surviving branch."""

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self._commands: deque[dict[str, object]] = deque()
        self._token_events = 0

    def feed_event(self, event: dict[str, object]) -> None:
        if event["type"] != "token_sampled":
            return
        self._token_events += 1
        if self._token_events == 1:
            self._commands.extend(
                [
                    {"type": "fork_at", "branch": 0, "width": 2},
                    {"type": "continue", "branch": 1},
                    {"type": "continue", "branch": 2},
                ]
            )
        elif self._token_events == 3:
            self._commands.extend(
                [
                    {"type": "finalize", "branch": 1},
                    {"type": "finalize", "branch": 2},
                    {"type": "kill", "branch": 0, "reason": "fork_replaced"},
                ]
            )

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands


class MeanRankingScheduler:
    """Finalize branches after ranking their externally supplied mean scores."""

    def __init__(
        self,
        config: dict[str, object],
        observed_scores: dict[int, list[float]],
    ) -> None:
        self.config = config
        self.observed_scores = observed_scores
        self._commands: deque[dict[str, object]] = deque()

    def feed_event(self, event: dict[str, object]) -> None:
        if event["type"] != "value_scored":
            return
        branch_id = int(event["branch"])
        self.observed_scores[branch_id].append(float(event["score"]))
        if branch_id == 0:
            self._commands.extend(
                [
                    {"type": "fork_at", "branch": 0, "width": 2},
                    {"type": "continue", "branch": 1},
                    {"type": "continue", "branch": 2},
                ]
            )
        elif branch_id == 2 and len(self.observed_scores[2]) == 1:
            self._commands.extend(
                [
                    {"type": "finalize", "branch": 1},
                    {"type": "continue", "branch": 2},
                ]
            )
        elif branch_id == 2 and len(self.observed_scores[2]) == 2:
            self._commands.append({"type": "continue", "branch": 2})
        elif branch_id == 2 and len(self.observed_scores[2]) == 3:
            self._commands.extend(
                [
                    {"type": "finalize", "branch": 2},
                    {"type": "kill", "branch": 0, "reason": "fork_replaced"},
                ]
            )

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands


class EosForkScheduler:
    """Fork non-terminal tokens, but finalize immediately when eos is explicit."""

    def __init__(
        self,
        config: dict[str, object],
        observed_events: list[dict[str, object]],
    ) -> None:
        self.config = config
        self.observed_events = observed_events
        self._commands: deque[dict[str, object]] = deque()

    def feed_event(self, event: dict[str, object]) -> None:
        self.observed_events.append(event)
        if event["type"] != "token_sampled":
            return
        if event.get("eos", False):
            self._commands.append({"type": "finalize", "branch": event["branch"]})
        else:
            self._commands.extend(
                [
                    {"type": "fork_at", "branch": event["branch"], "width": 2},
                    {"type": "finalize", "branch": 1},
                    {"type": "finalize", "branch": 2},
                    {"type": "kill", "branch": 0, "reason": "fork_replaced"},
                ]
            )

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands


def request(*, budget_tokens: int = 3) -> GenerationRequest:
    return GenerationRequest(
        model="tiny-engine-model",
        messages=(Message(role="user", content="branch once"),),
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        stop=(),
        seed=41,
        user=None,
        tree=TreeExecution(
            policy="beam",
            branches=2,
            budget_tokens=budget_tokens,
            scorer=None,
        ),
    )


async def collect(engine: TreeKVEngine, generation_request: GenerationRequest):
    return [event async for event in engine.generate(generation_request)]


@pytest.mark.parametrize(
    ("temperature", "top_p", "seed"),
    [
        pytest.param(0.5, 0.7, 11, id="nucleus-sampling"),
        pytest.param(2.0, 1.0, 29, id="non-unit-temperature"),
    ],
)
def test_sample_reports_unscaled_model_logprob(
    temperature: float,
    top_p: float,
    seed: int,
) -> None:
    logits = torch.tensor([3.0, 2.0, 1.0, -1.0])
    generation_request = replace(
        request(),
        temperature=temperature,
        top_p=top_p,
    )

    token_id, logprob = TreeKVEngine._sample(
        logits,
        generation_request,
        torch.Generator().manual_seed(seed),
    )

    expected = float(torch.log_softmax(logits.float(), dim=-1)[token_id].item())
    assert math.isfinite(logprob)
    assert logprob == pytest.approx(expected)


def test_null_seed_resolves_to_documented_zero_default() -> None:
    assert TreeKVEngine._resolve_seed(None) == 0
    assert TreeKVEngine._resolve_seed(17) == 17


def test_fork_ids_events_and_kill_reclaim_real_tree_kv_pages(tiny_engine_case) -> None:
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=ScriptedScheduler,
    )

    events = asyncio.run(collect(engine, request()))

    starts = [event for event in events if isinstance(event, BranchStarted)]
    assert [(event.branch_id, event.parent_id) for event in starts] == [
        ("branch-0", None),
        ("branch-1", "branch-0"),
        ("branch-2", "branch-0"),
    ]
    assert any(
        isinstance(event, BranchPruned) and event.branch_id == "branch-2"
        for event in events
    )
    assert any(
        branch_id == 2 and after < before
        for branch_id, before, after in tiny_engine_case.executor.prune_accounting
    ), "a scheduler Kill must immediately release the killed leaf's KV reference"
    assert tiny_engine_case.executor.batch_decode_calls == [(1, 2)]
    done = next(event for event in events if isinstance(event, GenerationDone))
    token_events = [event for event in events if isinstance(event, TokenGenerated)]
    assert done.usage.completion_tokens == len(token_events) == 3
    assert all(event.token_id is not None for event in token_events)
    assert done.tree_summary is not None
    assert done.tree_summary.kv_reuse_ratio > 1.0
    assert set(done.tree_summary.final_scores) == {
        "branch-0",
        "branch-1",
        "branch-2",
    }


def test_convergent_children_batch_dedup_merge_and_measure_step_costs(
    tiny_engine_case,
) -> None:
    executor = type(tiny_engine_case.executor)(
        replace(tiny_engine_case.executor.config, page_size=2),
        model=tiny_engine_case.executor.model,
    )
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=ConvergingScheduler,
        dedup_every_steps=2,
    )

    events = asyncio.run(collect(engine, request()))

    merges = [event for event in events if isinstance(event, BranchMerged)]
    assert [(event.branch_id, event.into_branch_id) for event in merges] == [
        ("branch-2", "branch-1")
    ]
    assert executor.batch_decode_calls == [(1, 2)]
    assert executor.dedup_calls == 1

    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.tree_summary is not None
    assert done.tree_summary.merged_count == 1
    assert done.tree_summary.pruned_count == 1
    assert done.tree_summary.kv_reuse_ratio == (
        done.counters.logical_tokens / done.counters.physical_tokens
    )
    assert done.counters.unique_tokens_per_step == (1, 1)
    assert done.counters.branch_tokens_per_step == (1, 2)
    assert asdict(done.counters)["unique_tokens_per_step"] == (1, 1)
    assert asdict(done.counters)["branch_tokens_per_step"] == (1, 2)


def test_same_seed_produces_identical_winning_completion(tiny_engine_case) -> None:
    def build() -> TreeKVEngine:
        return TreeKVEngine(
            model_id="tiny-engine-model",
            executor=tiny_engine_case.executor,
            tokenizer=tiny_engine_case.tokenizer,
            scheduler_factory=ScriptedScheduler,
        )

    first = asyncio.run(collect(build(), request()))
    second = asyncio.run(collect(build(), request()))

    first_done = next(event for event in first if isinstance(event, GenerationDone))
    second_done = next(event for event in second if isinstance(event, GenerationDone))
    assert first_done.text == second_done.text
    assert first_done.tree_summary == second_done.tree_summary


def test_winner_uses_scheduler_mean_score_across_fork(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_scores: dict[int, list[float]] = {0: [], 1: [], 2: []}
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=lambda config: MeanRankingScheduler(config, observed_scores),
    )
    samples = iter(
        [
            (10, -10.0),
            (11, -0.1),
            (12, -1.0),
            (13, -1.0),
            (14, -1.0),
        ]
    )
    monkeypatch.setattr(
        engine,
        "_sample",
        lambda _logits, _request, _generator: next(samples),
    )
    generation_request = replace(
        request(budget_tokens=5),
        tree=TreeExecution(
            policy="beam",
            branches=2,
            budget_tokens=5,
            scorer="external",
        ),
    )

    events = asyncio.run(collect(engine, generation_request))

    assert observed_scores[1][0] == pytest.approx((-10.0 - 0.1) / 2)
    assert observed_scores[2] == pytest.approx(
        [(-10.0 - 1.0) / 2, (-10.0 - 2.0) / 3, (-10.0 - 3.0) / 4]
    )
    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.branch_id == "branch-2"
    assert done.tree_summary is not None
    assert done.tree_summary.winner_branch_id == "branch-2"
    assert done.tree_summary.final_scores["branch-2"] == pytest.approx(-3.25)
    assert done.tree_summary.final_scores["branch-1"] == pytest.approx(-5.05)


def test_real_scheduler_never_exceeds_requested_tree_budget(tiny_engine_case) -> None:
    pytest.importorskip("autotree_scheduler")
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
    )

    events = asyncio.run(collect(engine, request(budget_tokens=3)))
    done = next(event for event in events if isinstance(event, GenerationDone))

    assert done.usage.completion_tokens == 3
    assert sum(done.tree_summary.tokens_spent_per_branch.values()) == 3


@pytest.mark.parametrize("policy", ["beam", "best_first", "mcts"])
def test_real_scheduler_keeps_engine_lifecycle_aligned_during_dedup(
    tiny_engine_case,
    policy: str,
) -> None:
    pytest.importorskip("autotree_scheduler")
    executor = type(tiny_engine_case.executor)(
        replace(tiny_engine_case.executor.config, page_size=2),
        model=tiny_engine_case.executor.model,
    )
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=executor,
        tokenizer=tiny_engine_case.tokenizer,
    )

    generation_request = replace(
        request(budget_tokens=6),
        tree=TreeExecution(
            policy=policy,
            branches=2,
            budget_tokens=6,
            scorer=None,
        ),
    )
    events = asyncio.run(collect(engine, generation_request))

    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.usage.completion_tokens <= 6
    assert done.tree_summary is not None
    assert done.tree_summary.branch_count > 1
    assert done.tree_summary.kv_reuse_ratio > 1.0
    if policy == "beam":
        assert done.tree_summary.merged_count >= 1


def test_eos_feeds_branch_exhausted_and_finishes_with_stop(
    tiny_engine_case,
) -> None:
    observed_events: list[dict[str, object]] = []
    expected_id = int(
        tiny_engine_case.executor.prefill([5, 6, 7, 8]).next_logits(0).argmax().item()
    )
    tiny_engine_case.tokenizer.eos_token_id = expected_id
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=lambda config: ExhaustionScheduler(config, observed_events),
    )

    events = asyncio.run(collect(engine, replace(request(), tree=None)))

    assert [event["type"] for event in observed_events] == [
        "token_sampled",
        "branch_exhausted",
    ]
    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.finish_reason == "stop"


def test_eos_token_is_not_forked_after_sampling(tiny_engine_case) -> None:
    observed_events: list[dict[str, object]] = []
    expected_id = int(
        tiny_engine_case.executor.prefill([5, 6, 7, 8]).next_logits(0).argmax().item()
    )
    tiny_engine_case.tokenizer.eos_token_id = expected_id
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=lambda config: EosForkScheduler(config, observed_events),
    )

    events = asyncio.run(collect(engine, request()))

    starts = [event for event in events if isinstance(event, BranchStarted)]
    assert [(event.branch_id, event.parent_id) for event in starts] == [
        ("branch-0", None)
    ]
    sampled = next(
        event for event in observed_events if event["type"] == "token_sampled"
    )
    assert sampled["eos"] is True
    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.branch_id == "branch-0"
    assert done.finish_reason == "stop"


def test_eos_does_not_send_value_after_scheduler_finalization(
    tiny_engine_case,
) -> None:
    class RejectLateValueScheduler(EosForkScheduler):
        def feed_event(self, event: dict[str, object]) -> None:
            if event["type"] == "value_scored":
                raise AssertionError("value_scored arrived after eos finalization")
            super().feed_event(event)

    observed_events: list[dict[str, object]] = []
    expected_id = int(
        tiny_engine_case.executor.prefill([5, 6, 7, 8]).next_logits(0).argmax().item()
    )
    tiny_engine_case.tokenizer.eos_token_id = expected_id
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=lambda config: RejectLateValueScheduler(
            config, observed_events
        ),
    )
    generation_request = replace(
        request(),
        tree=TreeExecution(
            policy="beam",
            branches=2,
            budget_tokens=3,
            scorer="external",
        ),
    )

    events = asyncio.run(collect(engine, generation_request))

    assert [event["type"] for event in observed_events] == ["token_sampled"]
    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.finish_reason == "stop"


def test_mid_decode_capacity_exhaustion_is_promoted_to_engine_error(
    tiny_engine_case,
) -> None:
    executor = type(tiny_engine_case.executor)(
        replace(tiny_engine_case.executor.config, capacity_pages=2),
        model=tiny_engine_case.executor.model,
    )
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=ContinueUntilCapacityScheduler,
    )
    generation_request = replace(request(), max_tokens=8, tree=None)

    with pytest.raises(KVCapacityExceededError) as raised:
        asyncio.run(collect(engine, generation_request))

    assert raised.value.phase == "decode"
    assert raised.value.required_pages == 1
    assert raised.value.available_pages == 0
    assert raised.value.capacity_pages == 2


def test_engine_device_and_dtype_reach_executor_config(monkeypatch) -> None:
    import autotree_core.engine.treekv as treekv_module

    captured: dict[str, object] = {}

    class CapturingExecutor:
        def __init__(self, config):
            captured["config"] = config
            self.config = config

    monkeypatch.setattr(treekv_module, "ModelExecutor", CapturingExecutor)
    monkeypatch.setattr(
        treekv_module.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, model_id: type("Cfg", (), {"n_positions": 64})()),
    )
    monkeypatch.setattr(
        treekv_module.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, model_id: object()),
    )

    TreeKVEngine(
        model_id="captured-model",
        scheduler_factory=ScriptedScheduler,
        device="cpu",
        dtype="bfloat16",
    )

    config = captured["config"]
    assert config.device == torch.device("cpu")
    assert config.dtype is torch.bfloat16


def test_engine_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="dtype must be one of"):
        TreeKVEngine(model_id="any-model", dtype="int8")
