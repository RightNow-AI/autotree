"""Real HuggingFace + Tree-KV execution driven by the Rust scheduler binding."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from transformers import AutoTokenizer

from autotree_core.modeling import ModelExecution, ModelExecutor, ModelExecutorConfig

from .protocol import (
    BranchPruned,
    BranchStarted,
    EngineCounters,
    EngineUsage,
    GenerationDone,
    GenerationRequest,
    ModelMetadata,
    TokenGenerated,
    TreeSummary,
)


class SchedulerBinding(Protocol):
    def feed_event(self, event: dict[str, object]) -> None: ...

    def poll_commands(self) -> list[dict[str, object]]: ...


SchedulerFactory = Callable[[dict[str, object]], SchedulerBinding]


@dataclass(frozen=True, slots=True)
class _KVSnapshot:
    logical_tokens: int
    physical_tokens: int

    @property
    def ratio(self) -> float:
        return self.logical_tokens / max(self.physical_tokens, 1)


class TreeKVEngine:
    """CPU-capable demo engine using real model weights and real Tree-KV pages."""

    def __init__(
        self,
        model_id: str = "gpt2",
        *,
        executor: ModelExecutor | None = None,
        tokenizer: Any | None = None,
        scheduler_factory: SchedulerFactory | None = None,
    ) -> None:
        self.executor = executor or ModelExecutor(ModelExecutorConfig(model_id=model_id))
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id)
        if scheduler_factory is None:
            try:
                from autotree_scheduler import Scheduler
            except ImportError as error:
                raise RuntimeError(
                    "TreeKVEngine requires the autotree-scheduler PyO3 wheel; "
                    "build and install it with maturin before selecting --engine treekv"
                ) from error
            scheduler_factory = Scheduler
        self._scheduler_factory = scheduler_factory
        self._metadata = ModelMetadata(
            id=model_id,
            engine="treekv",
            description=(
                "CPU Tree-KV demo engine using real HuggingFace model weights and "
                "the Rust branch scheduler; no GPU throughput claim is implied."
            ),
            real_model_weights=True,
            tree_policies=("beam", "best_first", "mcts"),
        )

    @property
    def model_metadata(self) -> ModelMetadata:
        return self._metadata

    async def generate(self, request: GenerationRequest):
        if request.model != self._metadata.id:
            raise ValueError(
                f"request model {request.model!r} does not match loaded model "
                f"{self._metadata.id!r}"
            )
        started_at = time.perf_counter()
        prompt_ids = self._encode_prompt(request)
        execution = self.executor.prefill(prompt_ids)
        scheduler = self._scheduler_factory(self._scheduler_config(request))
        generator = torch.Generator(device=self.executor.config.device).manual_seed(
            request.seed if request.seed is not None else 0
        )

        parents: dict[int, int | None] = {execution.root_id: None}
        own_text: dict[int, list[str]] = {execution.root_id: []}
        path_text: dict[int, str] = {execution.root_id: ""}
        token_counts: dict[int, int] = {execution.root_id: 0}
        scores: dict[int, float] = {execution.root_id: 0.0}
        active = {execution.root_id}
        finalized: set[int] = set()
        exhaustion_pending: set[int] = set()
        stopped: set[int] = set()
        commands: deque[dict[str, object]] = deque()
        completion_tokens = 0
        pruned_count = 0
        first_token_at: float | None = None
        best_snapshot = self._snapshot(execution)

        yield BranchStarted(branch_id="branch-0", parent_id=None)

        async def advance(branch_id: int) -> TokenGenerated:
            nonlocal completion_tokens, first_token_at, best_snapshot
            if branch_id not in active:
                raise RuntimeError(f"Continue targeted inactive branch {branch_id}")
            budget = request.tree.budget_tokens if request.tree else request.max_tokens
            if completion_tokens >= budget:
                raise RuntimeError("scheduler Continue exceeded the request token budget")

            logits = execution.next_logits(branch_id)
            token_id, logprob = self._sample(logits, request, generator)
            self.executor.decode(execution, branch_id, token_id)
            token = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            token_index = token_counts[branch_id]
            token_counts[branch_id] += 1
            own_text[branch_id].append(token)
            path_text[branch_id] += token
            scores[branch_id] += logprob
            completion_tokens += 1
            if self._token_exhausts_branch(token_id, path_text[branch_id], request):
                exhaustion_pending.add(branch_id)
                stopped.add(branch_id)
            if first_token_at is None:
                first_token_at = time.perf_counter()
            best_snapshot = self._better_snapshot(best_snapshot, self._snapshot(execution))

            scheduler.feed_event(
                {
                    "type": "token_sampled",
                    "branch": branch_id,
                    "token": token_id,
                    "logprob": logprob,
                }
            )
            if self._uses_external_scorer(request):
                scheduler.feed_event(
                    {
                        "type": "value_scored",
                        "branch": branch_id,
                        "score": scores[branch_id] / max(token_counts[branch_id], 1),
                    }
                )
            commands.extend(scheduler.poll_commands())
            await asyncio.sleep(0)
            return TokenGenerated(
                branch_id=self._branch_name(branch_id),
                token=token,
                token_index=token_index,
            )

        yield await advance(execution.root_id)

        while commands:
            command = commands.popleft()
            command_type = str(command.get("type"))
            branch_id = int(command["branch"])
            if command_type == "continue":
                if branch_id in exhaustion_pending:
                    exhaustion_pending.remove(branch_id)
                    scheduler.feed_event(
                        {"type": "branch_exhausted", "branch": branch_id}
                    )
                    commands.extend(scheduler.poll_commands())
                    continue
                yield await advance(branch_id)
                continue
            if command_type == "fork_at":
                if branch_id not in active:
                    raise RuntimeError(f"ForkAt targeted inactive branch {branch_id}")
                width = int(command["width"])
                children_are_exhausted = branch_id in exhaustion_pending
                exhaustion_pending.discard(branch_id)
                active.remove(branch_id)
                for _ in range(width):
                    expected_id = max(parents) + 1
                    child_id = self.executor.fork(execution, branch_id)
                    if child_id != expected_id:
                        raise RuntimeError(
                            "TreeState child allocation diverged from scheduler contract: "
                            f"expected {expected_id}, got {child_id}"
                        )
                    parents[child_id] = branch_id
                    own_text[child_id] = []
                    path_text[child_id] = path_text[branch_id]
                    token_counts[child_id] = 0
                    scores[child_id] = scores[branch_id]
                    active.add(child_id)
                    if children_are_exhausted:
                        exhaustion_pending.add(child_id)
                        stopped.add(child_id)
                    yield BranchStarted(
                        branch_id=self._branch_name(child_id),
                        parent_id=self._branch_name(branch_id),
                    )
                best_snapshot = self._better_snapshot(
                    best_snapshot, self._snapshot(execution)
                )
                continue
            if command_type == "kill":
                exhaustion_pending.discard(branch_id)
                if branch_id in active:
                    active.remove(branch_id)
                self.executor.prune(execution, branch_id)
                pruned_count += 1
                yield BranchPruned(
                    branch_id=self._branch_name(branch_id),
                    score=scores[branch_id],
                )
                continue
            if command_type == "finalize":
                exhaustion_pending.discard(branch_id)
                if branch_id in active:
                    active.remove(branch_id)
                finalized.add(branch_id)
                self.executor.prune(execution, branch_id)
                continue
            raise RuntimeError(f"unsupported scheduler command type {command_type!r}")

        if active:
            raise RuntimeError(
                "scheduler stopped issuing commands with active branches remaining: "
                f"{sorted(active)}"
            )
        if not finalized:
            raise RuntimeError("scheduler terminated without a finalized branch")

        winner = max(finalized, key=lambda branch: (scores[branch], -branch))
        for branch_id in sorted(finalized - {winner}):
            pruned_count += 1
            yield BranchPruned(
                branch_id=self._branch_name(branch_id),
                score=scores[branch_id],
            )

        ended_at = time.perf_counter()
        winner_path = self._path(winner, parents)
        winner_text = "".join(
            token for branch_id in winner_path for token in own_text[branch_id]
        )
        useful_tokens = sum(token_counts[branch_id] for branch_id in winner_path)
        summary = None
        if request.tree is not None:
            summary = TreeSummary(
                policy=request.tree.policy,
                branch_count=len(parents),
                pruned_count=pruned_count,
                merged_count=0,
                winner_branch_id=self._branch_name(winner),
                tokens_spent_per_branch={
                    self._branch_name(branch_id): token_counts[branch_id]
                    for branch_id in sorted(parents)
                },
                final_scores={
                    self._branch_name(branch_id): scores[branch_id]
                    for branch_id in sorted(parents)
                },
                scorer=request.tree.scorer,
                kv_reuse_ratio=best_snapshot.ratio,
            )
        yield GenerationDone(
            branch_id=self._branch_name(winner),
            text=winner_text,
            finish_reason="stop" if winner in stopped else "length",
            usage=EngineUsage(
                prompt_tokens=len(prompt_ids),
                completion_tokens=completion_tokens,
            ),
            counters=EngineCounters(
                logical_tokens=best_snapshot.logical_tokens,
                physical_tokens=best_snapshot.physical_tokens,
                useful_tokens=useful_tokens,
                elapsed_seconds=max(ended_at - started_at, 1e-9),
                ttft_seconds=max((first_token_at or ended_at) - started_at, 0.0),
            ),
            tree_summary=summary,
        )

    def _encode_prompt(self, request: GenerationRequest) -> list[int]:
        prompt = "\n".join(
            f"{message.role}: {message.content}" for message in request.messages
        )
        prompt = f"{prompt}\nassistant:"
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not token_ids:
            raise ValueError("tokenizer produced an empty prompt")
        return list(token_ids)

    @staticmethod
    def _branch_name(branch_id: int) -> str:
        return f"branch-{branch_id}"

    @staticmethod
    def _path(branch_id: int, parents: dict[int, int | None]) -> tuple[int, ...]:
        reversed_path: list[int] = []
        current: int | None = branch_id
        while current is not None:
            reversed_path.append(current)
            current = parents[current]
        return tuple(reversed(reversed_path))

    @staticmethod
    def _snapshot(execution: ModelExecution) -> _KVSnapshot:
        stats = execution.stats
        return _KVSnapshot(
            logical_tokens=stats.logical_tokens,
            physical_tokens=stats.physical_tokens,
        )

    @staticmethod
    def _better_snapshot(current: _KVSnapshot, candidate: _KVSnapshot) -> _KVSnapshot:
        return candidate if candidate.ratio > current.ratio else current

    @staticmethod
    def _uses_external_scorer(request: GenerationRequest) -> bool:
        return request.tree is not None and request.tree.scorer in {
            "external",
            "value_head",
        }

    def _token_exhausts_branch(
        self,
        token_id: int,
        text: str,
        request: GenerationRequest,
    ) -> bool:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            is_eos = token_id == eos_token_id
        elif isinstance(eos_token_id, (list, tuple, set, frozenset)):
            is_eos = token_id in eos_token_id
        else:
            is_eos = False
        return is_eos or any(stop and stop in text for stop in request.stop)

    @classmethod
    def _scheduler_config(cls, request: GenerationRequest) -> dict[str, object]:
        tree = request.tree
        policy = tree.policy.replace("_", "-") if tree else "beam"
        scorer = tree.scorer if tree and tree.scorer is not None else "logprob"
        if scorer not in {"logprob", "external", "value_head"}:
            raise ValueError(
                "TreeKVEngine scorer must be 'logprob', 'external', or 'value_head'"
            )
        branches = tree.branches if tree else 1
        return {
            "policy": policy,
            "branches": branches,
            "fork_width": branches,
            "fork_at_tokens": [1] if tree and branches > 1 else [],
            "max_depth": max(1, request.max_tokens),
            "budget_tokens": tree.budget_tokens if tree else request.max_tokens,
            "per_branch_token_budget": request.max_tokens,
            "seed": request.seed if request.seed is not None else 0,
            "scorer": scorer,
        }

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        request: GenerationRequest,
        generator: torch.Generator,
    ) -> tuple[int, float]:
        scores = logits.float()
        if request.temperature == 0:
            token_id = int(torch.argmax(scores).item())
        else:
            scores = scores / request.temperature
            probabilities = torch.softmax(scores, dim=-1)
            if request.top_p < 1.0:
                sorted_probabilities, sorted_indices = torch.sort(
                    probabilities, descending=True
                )
                cumulative = torch.cumsum(sorted_probabilities, dim=-1)
                remove = cumulative > request.top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
                probabilities = torch.zeros_like(probabilities).scatter(
                    0, sorted_indices, sorted_probabilities
                )
                probabilities = probabilities / probabilities.sum()
            token_id = int(
                torch.multinomial(probabilities, 1, generator=generator).item()
            )
        logprob = float(torch.log_softmax(scores, dim=-1)[token_id].item())
        if not math.isfinite(logprob):
            raise RuntimeError("model produced a non-finite sampled-token logprob")
        return token_id, logprob


__all__ = ["TreeKVEngine"]
