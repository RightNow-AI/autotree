"""AutoTree Python SDK public API."""

from .client import TreeClient
from .errors import (
    AutoTreeError,
    ExportError,
    SSEParseError,
    TraceInvariantError,
    TreeHTTPError,
)
from .models import (
    BranchMergedEvent,
    BranchPrunedEvent,
    BranchStartedEvent,
    ChatCompletionResponse,
    DoneEvent,
    EngineCounters,
    RolloutBatch,
    RolloutBranch,
    RolloutTree,
    TokenEvent,
    TreeCompletionResponse,
    TreeParameters,
    TreeSummary,
    Usage,
)
from .rollout import rollout
from .trace import TraceAssembler

__all__ = [
    "AutoTreeError",
    "BranchMergedEvent",
    "BranchPrunedEvent",
    "BranchStartedEvent",
    "ChatCompletionResponse",
    "DoneEvent",
    "EngineCounters",
    "ExportError",
    "RolloutBatch",
    "RolloutBranch",
    "RolloutTree",
    "SSEParseError",
    "TokenEvent",
    "TraceAssembler",
    "TraceInvariantError",
    "TreeClient",
    "TreeCompletionResponse",
    "TreeHTTPError",
    "TreeParameters",
    "TreeSummary",
    "Usage",
    "rollout",
]
