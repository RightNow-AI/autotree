"""Input honesty contracts shared by the figure pipeline."""

from __future__ import annotations

from typing import Any


class ProvenanceError(ValueError):
    """Raised when a figure input lacks usable provenance."""


def require_provenance(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the input provenance."""

    return payload.get("provenance")
