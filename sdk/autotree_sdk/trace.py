"""Invariant-enforcing assembly of streamed tree traces."""

from __future__ import annotations

from .errors import TraceInvariantError
from .models import (
    BranchMergedEvent,
    BranchPrunedEvent,
    BranchStartedEvent,
    DoneEvent,
    Prompt,
    RolloutBranch,
    RolloutTree,
    TokenEvent,
    TreeEvent,
)


class TraceAssembler:
    """Build a rollout tree while rejecting contract violations immediately."""

    def __init__(self, prompt: Prompt = "") -> None:
        self.prompt = prompt
        self._branches: dict[str, RolloutBranch] = {}
        self._done: DoneEvent | None = None
        self._token_count = 0

    @property
    def branches(self) -> tuple[RolloutBranch, ...]:
        return tuple(self._branches.values())

    def add(self, event: TreeEvent) -> None:
        if self._done is not None:
            raise TraceInvariantError(
                "event_after_done", f"received {event.type} after done"
            )
        if isinstance(event, BranchStartedEvent):
            self._start(event)
        elif isinstance(event, TokenEvent):
            self._token(event)
        elif isinstance(event, BranchPrunedEvent):
            branch = self._known_live(event.branch_id, event.type)
            branch.status = "pruned"
            branch.prune_reason = event.reason
        elif isinstance(event, BranchMergedEvent):
            branch = self._known_live(event.branch_id, event.type)
            if event.into_branch_id == event.branch_id:
                raise TraceInvariantError(
                    "invalid_merge_target", "branch cannot merge into itself", branch_id=event.branch_id
                )
            self._known_live(event.into_branch_id, event.type)
            branch.status = "merged"
            branch.merged_into = event.into_branch_id
        elif isinstance(event, DoneEvent):
            self._complete(event)

    def finish(self) -> RolloutTree:
        """Return the complete tree, or raise if the stream lacked ``done``."""

        if self._done is None:
            unterminated = ", ".join(self._branches) or "none"
            raise TraceInvariantError(
                "missing_done",
                f"stream ended before terminal done; started branches: {unterminated}",
            )
        return RolloutTree(
            prompt=self.prompt,
            branches=list(self._branches.values()),
            usage=self._done.usage,
            tree_summary=self._done.tree,
        )

    def _start(self, event: BranchStartedEvent) -> None:
        if event.branch_id in self._branches:
            raise TraceInvariantError(
                "duplicate_branch", "branch_started repeated", branch_id=event.branch_id
            )
        if event.parent_id is None:
            path = [event.branch_id]
        else:
            parent = self._branches.get(event.parent_id)
            if parent is None:
                raise TraceInvariantError(
                    "unknown_parent",
                    f"parent {event.parent_id!r} has not started",
                    branch_id=event.branch_id,
                )
            path = [*parent.branch_path, event.branch_id]
        self._branches[event.branch_id] = RolloutBranch(
            branch_id=event.branch_id,
            parent_id=event.parent_id,
            branch_path=path,
        )

    def _token(self, event: TokenEvent) -> None:
        branch = self._known_live(event.branch_id, event.type)
        if branch.token_indices and event.token_index <= branch.token_indices[-1]:
            raise TraceInvariantError(
                "non_increasing_token_index",
                f"received {event.token_index} after {branch.token_indices[-1]}",
                branch_id=event.branch_id,
            )
        branch.token_indices.append(event.token_index)
        branch.tokens.append(event.token)
        # The current wire event has no token_id. Preserve alignment without
        # inventing a tokenizer- or model-dependent ID.
        branch.token_ids.append(None)
        branch.token_logprobs.append(event.logprob)
        self._token_count += 1

    def _known_live(self, branch_id: str, event_type: str) -> RolloutBranch:
        branch = self._branches.get(branch_id)
        if branch is None:
            raise TraceInvariantError(
                "unknown_branch",
                f"{event_type} references a branch that has not started",
                branch_id=branch_id,
            )
        if branch.status != "live":
            raise TraceInvariantError(
                "branch_not_live",
                f"{event_type} arrived after terminal status {branch.status}",
                branch_id=branch_id,
            )
        return branch

    def _complete(self, event: DoneEvent) -> None:
        if event.usage.completion_tokens != self._token_count:
            raise TraceInvariantError(
                "usage_token_mismatch",
                f"usage.completion_tokens={event.usage.completion_tokens}, "
                f"streamed_token_events={self._token_count}",
            )
        if event.tree.branch_count != len(self._branches):
            raise TraceInvariantError(
                "branch_count_mismatch",
                f"summary={event.tree.branch_count}, observed={len(self._branches)}",
            )
        pruned_count = sum(branch.pruned for branch in self._branches.values())
        if event.tree.pruned_count != pruned_count:
            raise TraceInvariantError(
                "pruned_count_mismatch",
                f"summary={event.tree.pruned_count}, observed={pruned_count}",
            )
        observed_tokens = {
            branch_id: len(branch.tokens)
            for branch_id, branch in self._branches.items()
        }
        if event.tree.tokens_spent_per_branch != observed_tokens:
            raise TraceInvariantError(
                "branch_token_count_mismatch",
                f"summary={event.tree.tokens_spent_per_branch}, observed={observed_tokens}",
            )
        for branch in self._branches.values():
            if branch.status == "live":
                branch.status = "completed"
        self._done = event
