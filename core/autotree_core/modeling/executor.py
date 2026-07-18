"""Real HuggingFace causal-LM execution backed by paged Tree-KV storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, DynamicCache, PreTrainedModel

from autotree_core.kv import KVPoolConfig, KVStats, PagedKVPool, TreeState

from .config import ModelExecutorConfig


def _cache_pairs(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Normalize supported Transformers cache representations."""
    if hasattr(cache, "layers"):
        pairs = tuple((layer.keys, layer.values) for layer in cache.layers)
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        pairs = tuple(zip(cache.key_cache, cache.value_cache, strict=True))
    elif isinstance(cache, (tuple, list)):
        pairs = tuple((layer[0], layer[1]) for layer in cache)
    else:
        raise TypeError(f"unsupported HuggingFace cache type: {type(cache).__name__}")

    if not pairs or any(k is None or v is None for k, v in pairs):
        raise ValueError("the model returned an incomplete KV cache")
    return pairs


def _pool_layout(cache: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a batch-one HF cache to ``[layers, tokens, heads, dim]``."""
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for key, value in _cache_pairs(cache):
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise ValueError("model KV tensors must match [batch, heads, tokens, dim]")
        if key.shape[0] != 1:
            raise ValueError("ModelExecutor currently accepts one prompt per execution")
        keys.append(key[0].transpose(0, 1).detach().contiguous())
        values.append(value[0].transpose(0, 1).detach().contiguous())
    return torch.stack(keys), torch.stack(values)


def _dynamic_cache(
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    model_config: Any,
) -> DynamicCache:
    """Build a mutable HF cache across the supported Transformers APIs."""
    materialized = tuple(pairs)
    try:
        return DynamicCache(materialized, config=model_config)
    except TypeError:
        cache = DynamicCache()
        for layer_idx, (key, value) in enumerate(materialized):
            cache.update(key, value, layer_idx)
        return cache


@dataclass(frozen=True, slots=True)
class DecodeOutput:
    """The distribution after appending one token to a branch."""

    branch_id: int
    token_id: int
    logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    """Generated ids plus the live Tree-KV execution that produced them."""

    sequences: torch.Tensor
    generated_ids: tuple[int, ...]
    branch_id: int
    execution: ModelExecution


@dataclass(slots=True)
class ModelExecution:
    """One prompt's live branch topology and paged model KV state."""

    pool: PagedKVPool
    tree: TreeState
    _owner: object = field(repr=False)
    _next_logits: dict[int, torch.Tensor] = field(repr=False)
    _token_ids: dict[int, list[int]] = field(repr=False)

    @property
    def root_id(self) -> int:
        return self.tree.root_id

    @property
    def stats(self) -> KVStats:
        return self.pool.stats

    @property
    def branch_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.tree.branches))

    def next_logits(self, branch_id: int) -> torch.Tensor:
        """Return a detached copy of a branch's next-token logits."""
        try:
            return self._next_logits[branch_id].clone()
        except KeyError:
            raise KeyError(branch_id) from None

    def token_ids(self, branch_id: int) -> tuple[int, ...]:
        """Return the root-to-node token ids for a live branch."""
        try:
            return tuple(self._token_ids[branch_id])
        except KeyError:
            raise KeyError(branch_id) from None

    def gather_kv(
        self, branch_id: int, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize one layer of a branch from its physical pages."""
        return self.tree.gather(branch_id, layer)

    def attention_metadata(
        self, branch_ids: Sequence[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build kernel-ready int32 block tables and context lengths."""
        selected = self.branch_ids if branch_ids is None else tuple(branch_ids)
        branches = [self.tree.get_branch(branch_id) for branch_id in selected]
        max_pages = max((len(branch.block_table) for branch in branches), default=0)
        tables = torch.full(
            (len(branches), max_pages),
            -1,
            dtype=torch.int32,
            device=self.pool.config.device,
        )
        for row, branch in enumerate(branches):
            if branch.block_table:
                tables[row, : len(branch.block_table)] = torch.tensor(
                    branch.block_table,
                    dtype=torch.int32,
                    device=self.pool.config.device,
                )
        lengths = torch.tensor(
            [branch.num_tokens for branch in branches],
            dtype=torch.int32,
            device=self.pool.config.device,
        )
        return tables, lengths


class ModelExecutor:
    """Execute a HuggingFace causal LM while TreeState owns every KV byte."""

    def __init__(
        self,
        config: ModelExecutorConfig | None = None,
        *,
        model: PreTrainedModel | None = None,
    ) -> None:
        self.config = config or ModelExecutorConfig()
        self._owner = object()
        self.model = model if model is not None else self._load_model()
        self.model.to(device=self.config.device, dtype=self.config.dtype)
        self.model.eval()
        self.model.requires_grad_(False)

        model_config = self.model.config
        self.num_layers = int(model_config.num_hidden_layers)
        self.num_attention_heads = int(model_config.num_attention_heads)
        self.num_kv_heads = int(
            getattr(model_config, "num_key_value_heads", None)
            or self.num_attention_heads
        )
        hidden_size = int(model_config.hidden_size)
        self.head_dim = int(
            getattr(model_config, "head_dim", None)
            or hidden_size // self.num_attention_heads
        )
        if hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "model hidden size must divide evenly across attention heads"
            )

    def _load_model(self) -> PreTrainedModel:
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
            "dtype": self.config.dtype,
            "attn_implementation": self.config.attn_implementation,
        }
        if self.config.revision is not None:
            kwargs["revision"] = self.config.revision
        return AutoModelForCausalLM.from_pretrained(self.config.model_id, **kwargs)

    def prefill(self, input_ids: torch.Tensor | Sequence[int]) -> ModelExecution:
        """Run real model prefill and place every layer's K/V in Tree-KV pages."""
        ids = self._normalize_input_ids(input_ids)
        attention_mask = torch.ones_like(ids)
        with torch.inference_mode():
            output = self.model(
                input_ids=ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        keys, values = _pool_layout(output.past_key_values)
        self._validate_cache_shape(keys, expected_tokens=ids.shape[1])

        required_pages = ceil(ids.shape[1] / self.config.page_size)
        if required_pages > self.config.capacity_pages:
            raise ValueError(
                "prompt requires more KV pages than capacity_pages provides: "
                f"{required_pages} > {self.config.capacity_pages}"
            )
        pool = PagedKVPool(
            KVPoolConfig(
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                capacity=self.config.capacity_pages,
                page_size=self.config.page_size,
                dtype=self.config.dtype,
                device=self.config.device,
            )
        )
        tree = TreeState(pool)
        tree.append_tokens(tree.root_id, keys, values)
        logits = output.logits[0, -1].detach().contiguous()
        return ModelExecution(
            pool=pool,
            tree=tree,
            _owner=self._owner,
            _next_logits={tree.root_id: logits},
            _token_ids={tree.root_id: ids[0].tolist()},
        )

    def decode(
        self,
        execution: ModelExecution,
        branch_id: int,
        token_id: int,
    ) -> DecodeOutput:
        """Append one token using model attention over KV gathered from pages."""
        self._require_execution(execution)
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token_id must be an integer")
        branch = execution.tree.get_branch(branch_id)
        cache_pairs = []
        for layer in range(self.num_layers):
            key, value = execution.gather_kv(branch_id, layer)
            cache_pairs.append(
                (
                    key.transpose(0, 1).unsqueeze(0),
                    value.transpose(0, 1).unsqueeze(0),
                )
            )
        cache = _dynamic_cache(cache_pairs, self.model.config)
        input_ids = torch.tensor(
            [[token_id]], dtype=torch.long, device=self.config.device
        )
        attention_mask = torch.ones(
            (1, branch.num_tokens + 1),
            dtype=torch.long,
            device=self.config.device,
        )
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        keys, values = _pool_layout(output.past_key_values)
        self._validate_cache_shape(keys, expected_tokens=branch.num_tokens + 1)
        execution.tree.append_token(branch_id, keys[:, -1], values[:, -1])
        logits = output.logits[0, -1].detach().contiguous()
        execution._next_logits[branch_id] = logits
        execution._token_ids[branch_id].append(token_id)
        return DecodeOutput(
            branch_id=branch_id, token_id=token_id, logits=logits.clone()
        )

    def fork(self, execution: ModelExecution, branch_id: int) -> int:
        """Fork a branch by sharing its Tree-KV pages until a COW write."""
        self._require_execution(execution)
        new_branch_id = execution.tree.fork(branch_id)
        execution._next_logits[new_branch_id] = execution._next_logits[
            branch_id
        ].clone()
        execution._token_ids[new_branch_id] = execution._token_ids[branch_id].copy()
        return new_branch_id

    def prune(self, execution: ModelExecution, branch_id: int) -> None:
        """Prune a leaf and immediately reclaim unreferenced KV pages."""
        self._require_execution(execution)
        execution.tree.prune(branch_id)
        del execution._next_logits[branch_id]
        del execution._token_ids[branch_id]

    def deduplicate(self, execution: ModelExecution) -> int:
        """Deduplicate byte-identical full pages across live branches."""
        self._require_execution(execution)
        return execution.tree.dedup_scan()

    def sample_next_token(
        self,
        logits: torch.Tensor,
        *,
        generator: torch.Generator,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> int:
        """Sample one token with a caller-owned deterministic generator."""
        if logits.ndim != 1:
            raise ValueError("logits must have shape [vocab_size]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        scores = logits.float() / temperature
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k must be a positive integer or None")
            top_k = min(top_k, scores.numel())
            threshold = torch.topk(scores, top_k).values[-1]
            scores = scores.masked_fill(scores < threshold, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        return int(torch.multinomial(probabilities, 1, generator=generator).item())

    def generate(
        self,
        input_ids: torch.Tensor | Sequence[int],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        seed: int = 0,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> GenerationOutput:
        """Generate greedily or with seeded sampling through one Tree-KV branch."""
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer")
        execution = self.prefill(input_ids)
        branch_id = execution.root_id
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            logits = execution._next_logits[branch_id]
            if do_sample:
                token_id = self.sample_next_token(
                    logits,
                    generator=generator,
                    temperature=temperature,
                    top_k=top_k,
                )
            else:
                token_id = int(torch.argmax(logits).item())
            generated.append(token_id)
            self.decode(execution, branch_id, token_id)

        sequences = torch.tensor(
            [execution._token_ids[branch_id]],
            dtype=torch.long,
            device=self.config.device,
        )
        return GenerationOutput(
            sequences=sequences,
            generated_ids=tuple(generated),
            branch_id=branch_id,
            execution=execution,
        )

    def _normalize_input_ids(
        self, input_ids: torch.Tensor | Sequence[int]
    ) -> torch.Tensor:
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.config.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("input_ids must contain one non-empty token sequence")
        return ids.contiguous()

    def _validate_cache_shape(
        self, keys: torch.Tensor, *, expected_tokens: int
    ) -> None:
        expected = (
            self.num_layers,
            expected_tokens,
            self.num_kv_heads,
            self.head_dim,
        )
        if tuple(keys.shape) != expected:
            raise ValueError(
                f"model returned KV shape {tuple(keys.shape)}, expected {expected}"
            )
        if keys.dtype is not self.config.dtype or keys.device != self.config.device:
            raise ValueError("model KV dtype/device does not match the executor config")

    def _require_execution(self, execution: ModelExecution) -> None:
        if (
            not isinstance(execution, ModelExecution)
            or execution._owner is not self._owner
        ):
            raise ValueError("execution was not created by this ModelExecutor")


__all__ = [
    "DecodeOutput",
    "GenerationOutput",
    "ModelExecution",
    "ModelExecutor",
]
