"""Typed SDK errors with stable machine-readable violation names."""


class AutoTreeError(Exception):
    """Base class for SDK errors."""


class TreeHTTPError(AutoTreeError):
    """An HTTP request failed; POST requests are never retried implicitly."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class SSEParseError(AutoTreeError):
    """An SSE frame or typed event could not be parsed."""

    def __init__(self, violation: str, detail: str) -> None:
        self.violation = violation
        self.detail = detail
        super().__init__(f"{violation}: {detail}")

class TraceInvariantError(AutoTreeError):
    """A streamed trace violated the tree wire contract."""

    def __init__(
        self, violation: str, detail: str = "", *, branch_id: str | None = None
    ) -> None:
        self.violation = violation
        self.detail = detail
        self.branch_id = branch_id
        message = violation
        if branch_id is not None:
            message += f" [branch={branch_id}]"
        if detail:
            message += f": {detail}"
        super().__init__(message)


class ExportError(AutoTreeError):
    """A rollout cannot be represented honestly in the requested export."""

    def __init__(self, violation: str, detail: str) -> None:
        self.violation = violation
        self.detail = detail
        super().__init__(f"{violation}: {detail}")
