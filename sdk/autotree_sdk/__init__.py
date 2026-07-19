"""AutoTree Python SDK public API."""

from .client import TreeClient
from .errors import (
    AutoTreeError,
    ExportError,
    SSEParseError,
    TraceInvariantError,
    TreeHTTPError,
    TreeStreamError,
)
from .models import (
    BranchMergedEvent,
    BranchPrunedEvent,
    BranchStartedEvent,
    ChatCompletionResponse,
    DoneEvent,
    EngineCounters,
    ErrorEvent,
    RolloutBatch,
    RolloutBranch,
    RolloutTree,
    StreamErrorDetails,
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
    "ErrorEvent",
    "ExportError",
    "RolloutBatch",
    "RolloutBranch",
    "RolloutTree",
    "SSEParseError",
    "StreamErrorDetails",
    "TokenEvent",
    "TraceAssembler",
    "TraceInvariantError",
    "TreeClient",
    "TreeCompletionResponse",
    "TreeHTTPError",
    "TreeParameters",
    "TreeStreamError",
    "TreeSummary",
    "Usage",
    "rollout",
]
