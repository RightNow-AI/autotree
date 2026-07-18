from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from autotree_core.modeling import ModelExecutor, ModelExecutorConfig


@dataclass(frozen=True, slots=True)
class ModelCase:
    name: str
    model_id: str
    executor: ModelExecutor
    prompt: tuple[int, ...]


def _write_tiny_model(path: Path) -> str:
    torch.manual_seed(314159)
    config = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_ctx=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        use_cache=True,
    )
    model = GPT2LMHeadModel(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture(scope="session")
def tiny_model_id(tmp_path_factory: pytest.TempPathFactory) -> str:
    return _write_tiny_model(tmp_path_factory.mktemp("tiny-gpt2"))


@pytest.fixture(
    scope="session",
    params=(
        pytest.param("tiny", id="tiny-random-gpt2"),
        pytest.param("gpt2", id="gpt2-124m"),
    ),
)
def model_case(request: pytest.FixtureRequest, tiny_model_id: str) -> ModelCase:
    name = str(request.param)
    model_id = tiny_model_id if name == "tiny" else "gpt2"
    config = ModelExecutorConfig(
        model_id=model_id,
        device="cpu",
        dtype=torch.float32,
        page_size=4,
        capacity_pages=32,
        local_files_only=name == "tiny",
    )
    try:
        executor = ModelExecutor(config)
    except OSError as error:
        if name == "gpt2":
            pytest.fail(f"required gpt2 download/load failed: {error}")
        raise
    return ModelCase(
        name=name,
        model_id=model_id,
        executor=executor,
        prompt=(5, 6, 7, 8, 9, 10, 11, 12),
    )
