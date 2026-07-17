"""Public API for AutoTree's paged KV cache."""

from .config import PAGE_SIZE, KVPoolConfig
from .errors import KVCapacityError, KVError, KVInvariantError
from .pool import KVStats, PagedKVPool

__all__ = [
    "PAGE_SIZE",
    "KVCapacityError",
    "KVError",
    "KVInvariantError",
    "KVPoolConfig",
    "KVStats",
    "PagedKVPool",
]
