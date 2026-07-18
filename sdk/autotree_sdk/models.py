"""Typed request, response, event, and rollout models."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .errors import ExportError, SSEParseError

TreePolicy: TypeAlias = Literal["beam", "best_first", "mcts"]
Prompt: TypeAlias = str | list[dict[str, Any]]
FinalScores: TypeAlias = dict[str, float] | list[float]


class TreeParameters(BaseModel):
    """Tree-search controls accepted by AutoTree serving endpoints."""

    policy: TreePolicy = "beam"
    branches: int = Field(gt=0)
    budget_tokens: int = Field(gt=0)
    scorer: str | None = None


class Usage(BaseModel):
    """OpenAI-compatible token usage returned by the server."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class TreeSummary(BaseModel):
    """Server summary for a completed tree execution.

    The wire contract does not state whether ``final_scores`` is branch-keyed
    or positional, so both JSON shapes are accepted. Preference export refuses
    positional scores because branch identity cannot be inferred safely.
    """

    model_config = ConfigDict(extra="allow")

    branch_count: int = Field(ge=0)
    pruned_count: int = Field(ge=0)
    tokens_spent_per_branch: dict[str, int]
    final_scores: FinalScores


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None


class CompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """Typed subset of a non-stream OpenAI chat completion response."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    object: str | None = None
    created: int | None = None
    model: str | None = None
    choices: list[CompletionChoice]
    usage: Usage | None = None
    tree: TreeSummary | dict[str, Any] | None = None


class TreeCompletionResponse(ChatCompletionResponse):
    """Winning completion plus the required tree summary."""

    tree: TreeSummary


class BranchStartedEvent(BaseModel):
    type: Literal["branch_started"] = "branch_started"
    branch_id: str
    parent_id: str | None = None


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    branch_id: str
    token_index: int = Field(ge=0)
    token: str
    logprob: float


class BranchPrunedEvent(BaseModel):
    type: Literal["branch_pruned"] = "branch_pruned"
    branch_id: str
    reason: str


class BranchMergedEvent(BaseModel):
    type: Literal["branch_merged"] = "branch_merged"
    branch_id: str
    into_branch_id: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    usage: Usage
    tree: TreeSummary


TreeEvent: TypeAlias = Annotated[
    BranchStartedEvent
    | TokenEvent
    | BranchPrunedEvent
    | BranchMergedEvent
    | DoneEvent,
    Field(discriminator="type"),
]
_TREE_EVENT_ADAPTER = TypeAdapter(TreeEvent)


def parse_tree_event(payload: Any) -> TreeEvent:
    """Parse one SSE JSON object into its concrete typed event."""

    try:
        return _TREE_EVENT_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise SSEParseError("invalid_tree_event", str(exc)) from exc


@dataclass(slots=True)
class RolloutBranch:
    """One branch reconstructed from streamed token events."""

    branch_id: str
    parent_id: str | None
    branch_path: list[str]
    tokens: list[str] = field(default_factory=list)
    token_ids: list[int | None] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    token_indices: list[int] = field(default_factory=list)
    status: Literal["live", "completed", "pruned", "merged"] = "live"
    prune_reason: str | None = None
    merged_into: str | None = None

    @property
    def completion(self) -> str:
        return "".join(self.tokens)

    @property
    def pruned(self) -> bool:
        return self.status == "pruned"

    @property
    def cumulative_logprob(self) -> float:
        return sum(self.token_logprobs)

    def as_sample(self, prompt: Prompt, prompt_index: int) -> dict[str, Any]:
        """Return a flat, JSON-friendly sample shared by rollout exporters."""

        return {
            "prompt": prompt,
            "completion": self.completion,
            "token_ids": list(self.token_ids),
            "token_indices": list(self.token_indices),
            "token_logprobs": list(self.token_logprobs),
            "cumulative_logprob": self.cumulative_logprob,
            "branch_path": list(self.branch_path),
            "branch_id": self.branch_id,
            "parent_id": self.parent_id,
            "prompt_index": prompt_index,
            "status": self.status,
            "pruned": self.pruned,
            "prune_reason": self.prune_reason,
            "merged_into": self.merged_into,
        }


@dataclass(slots=True)
class RolloutTree:
    """A complete branching trace for one input prompt."""

    prompt: Prompt
    branches: list[RolloutBranch]
    usage: Usage
    tree_summary: TreeSummary

    def branch(self, branch_id: str) -> RolloutBranch:
        for branch in self.branches:
            if branch.branch_id == branch_id:
                return branch
        raise KeyError(branch_id)


@dataclass(slots=True)
class RolloutBatch:
    """Rollout trees plus common RL post-training export adapters."""

    trees: list[RolloutTree]

    def to_grpo_samples(
        self, *, include_pruned: bool = False, include_merged: bool = False
    ) -> list[dict[str, Any]]:
        """Export flat prompt/completion samples for grouped-policy training.

        Each record retains the original prompt (text or chat messages), token
        logprobs, cumulative logprob, root-to-branch path, and pruning metadata.
        ``prompt_index`` is the group key. By default only live-at-done branches
        are emitted; diagnostic pruned/merged traces are opt-in.
        """

        samples: list[dict[str, Any]] = []
        for prompt_index, tree in enumerate(self.trees):
            for branch in tree.branches:
                if branch.status == "pruned" and not include_pruned:
                    continue
                if branch.status == "merged" and not include_merged:
                    continue
                samples.append(branch.as_sample(tree.prompt, prompt_index))
        return samples

    def to_rlhf_pairs(self, *, include_pruned: bool = True) -> list[dict[str, Any]]:
        """Export score-ordered chosen/rejected preference pairs.

        ``final_scores`` must map branch IDs to numeric scores. Each unequal
        within-prompt pair becomes one record with full chosen/rejected samples
        and their scores. Merged branches are excluded because they do not own
        an independent terminal completion.
        """

        pairs: list[dict[str, Any]] = []
        for prompt_index, tree in enumerate(self.trees):
            scores = tree.tree_summary.final_scores
            if not isinstance(scores, dict):
                raise ExportError(
                    "ambiguous_final_scores",
                    "RLHF pairs require final_scores keyed by branch ID; "
                    "the server returned a positional list",
                )
            candidates = [
                branch
                for branch in tree.branches
                if branch.status != "merged" and (include_pruned or not branch.pruned)
            ]
            missing = [
                branch.branch_id
                for branch in candidates
                if branch.branch_id not in scores
            ]
            if missing:
                raise ExportError(
                    "missing_branch_scores",
                    f"final_scores lacks branch IDs: {', '.join(sorted(missing))}",
                )
            for left, right in combinations(candidates, 2):
                left_score = scores[left.branch_id]
                right_score = scores[right.branch_id]
                if left_score == right_score:
                    continue
                chosen, rejected = (
                    (left, right) if left_score > right_score else (right, left)
                )
                pairs.append(
                    {
                        "prompt": tree.prompt,
                        "prompt_index": prompt_index,
                        "chosen": chosen.as_sample(tree.prompt, prompt_index),
                        "rejected": rejected.as_sample(tree.prompt, prompt_index),
                        "chosen_score": scores[chosen.branch_id],
                        "rejected_score": scores[rejected.branch_id],
                    }
                )
        return pairs
