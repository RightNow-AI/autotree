"""HTTP request validation models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = Field(min_length=1)
    content: str


class TreeParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Literal["beam", "best_first", "mcts"]
    branches: int = Field(ge=1, le=64)
    budget_tokens: int = Field(ge=1, le=1_000_000)
    scorer: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    max_tokens: int = Field(default=16, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    seed: int | None = None
    tree: TreeParameters | None = None


class TreeCompletionRequest(ChatCompletionRequest):
    tree: TreeParameters
