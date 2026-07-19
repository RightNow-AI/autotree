"""Shared engine request/event contract consumed by autotree-serve."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import InitVar, asdict, dataclass, field
from typing import Literal, Protocol, TypeAlias, runtime_checkable
from weakref import finalize


_STEP_COSTS: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
_STEP_COST_FINALIZERS: dict[int, finalize] = {}


def _drop_step_costs(counter_id: int) -> None:
    _STEP_COSTS.pop(counter_id, None)
    _STEP_COST_FINALIZERS.pop(counter_id, None)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    id: str
    engine: str
    description: str
    real_model_weights: bool
    tree_policies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class TreeExecution:
    policy: Literal["beam", "best_first", "mcts"]
    branches: int
    budget_tokens: int
    scorer: str | None


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    model: str
    messages: tuple[Message, ...]
    max_tokens: int
    temperature: float
    top_p: float
    stop: tuple[str, ...]
    seed: int | None
    user: str | None
    tree: TreeExecution | None


class KVCapacityExceededError(RuntimeError):
    """A foreseeable Tree-KV capacity limit at the engine boundary."""

    def __init__(
        self,
        *,
        phase: Literal["admission", "decode"],
        required_pages: int,
        available_pages: int,
        capacity_pages: int,
    ) -> None:
        self.phase = phase
        self.required_pages = required_pages
        self.available_pages = available_pages
        self.capacity_pages = capacity_pages
        super().__init__(
            f"Tree-KV {phase} requires {required_pages} page(s), but only "
            f"{available_pages} are available within the {capacity_pages}-page limit. "
            "Increase --kv-pages or reduce prompt/tree size."
        )


@dataclass(frozen=True, slots=True)
class BranchStarted:
    branch_id: str
    parent_id: str | None
    type: Literal["branch_started"] = field(default="branch_started", init=False)


@dataclass(frozen=True, slots=True)
class TokenGenerated:
    branch_id: str
    token: str
    token_index: int
    logprob: float
    type: Literal["token"] = field(default="token", init=False)


@dataclass(frozen=True, slots=True)
class BranchPruned:
    branch_id: str
    reason: str
    type: Literal["branch_pruned"] = field(default="branch_pruned", init=False)


@dataclass(frozen=True, slots=True)
class BranchMerged:
    branch_id: str
    into_branch_id: str
    type: Literal["branch_merged"] = field(default="branch_merged", init=False)


@dataclass(frozen=True, slots=True)
class EngineUsage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class TreeSummary:
    policy: str
    branch_count: int
    pruned_count: int
    merged_count: int
    winner_branch_id: str
    tokens_spent_per_branch: dict[str, int]
    final_scores: dict[str, float]
    scorer: str | None
    kv_reuse_ratio: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class EngineCounters:
    logical_tokens: int
    physical_tokens: int
    useful_tokens: int
    elapsed_seconds: float
    ttft_seconds: float
    _unique_tokens_per_step: InitVar[tuple[int, ...]] = ()
    _branch_tokens_per_step: InitVar[tuple[int, ...]] = ()

    def __post_init__(
        self,
        _unique_tokens_per_step: tuple[int, ...],
        _branch_tokens_per_step: tuple[int, ...],
    ) -> None:
        unique = tuple(_unique_tokens_per_step)
        branch = tuple(_branch_tokens_per_step)
        if len(unique) != len(branch):
            raise ValueError("step token counters must have matching lengths")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*unique, *branch)
        ):
            raise ValueError("step token counters must contain non-negative integers")
        if any(
            unique_count > branch_count
            for unique_count, branch_count in zip(unique, branch, strict=True)
        ):
            raise ValueError("unique step tokens cannot exceed branch step tokens")
        counter_id = id(self)
        _STEP_COSTS[counter_id] = (unique, branch)
        _STEP_COST_FINALIZERS[counter_id] = finalize(self, _drop_step_costs, counter_id)

    @property
    def unique_tokens_per_step(self) -> tuple[int, ...]:
        """Measured distinct logical token paths produced by each decode forward."""
        return _STEP_COSTS[id(self)][0]

    @property
    def branch_tokens_per_step(self) -> tuple[int, ...]:
        """Measured sum-over-branches token count for each decode forward."""
        return _STEP_COSTS[id(self)][1]


@dataclass(frozen=True, slots=True)
class GenerationDone:
    branch_id: str
    text: str
    finish_reason: Literal["stop", "length"]
    usage: EngineUsage
    counters: EngineCounters
    tree_summary: TreeSummary | None
    type: Literal["done"] = field(default="done", init=False)


EngineEvent: TypeAlias = (
    BranchStarted | TokenGenerated | BranchPruned | BranchMerged | GenerationDone
)


@runtime_checkable
class EngineProtocol(Protocol):
    @property
    def model_metadata(self) -> ModelMetadata: ...

    def generate(self, request: GenerationRequest) -> AsyncIterator[EngineEvent]: ...


__all__ = [
    "BranchMerged",
    "BranchPruned",
    "BranchStarted",
    "EngineCounters",
    "EngineEvent",
    "EngineProtocol",
    "EngineUsage",
    "GenerationDone",
    "GenerationRequest",
    "KVCapacityExceededError",
    "Message",
    "ModelMetadata",
    "TokenGenerated",
    "TreeExecution",
    "TreeSummary",
]
