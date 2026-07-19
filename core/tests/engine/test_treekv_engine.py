from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, replace

import pytest

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
