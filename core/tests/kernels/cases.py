"""Seeded tree-layout builders and an independent dense SDPA oracle."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TreeAttentionCase:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    dense_k: tuple[torch.Tensor, ...]
    dense_v: tuple[torch.Tensor, ...]


def _random_tokens(
    length: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (length, num_kv_heads, head_dim)
    k = torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)
    v = torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)
    return k, v


def _page_key(k_page: torch.Tensor, v_page: torch.Tensor) -> tuple[bytes, bytes]:
    return (
        k_page.contiguous().view(torch.uint8).numpy().tobytes(),
        v_page.contiguous().view(torch.uint8).numpy().tobytes(),
    )


def build_random_tree_case(
    *,
    seed: int,
    num_branches: int,
    gqa_ratio: int,
    context_remainder: int,
    dtype: torch.dtype,
    page_size: int = 16,
    num_kv_heads: int = 2,
    head_dim: int = 8,
) -> TreeAttentionCase:
    """Build branch paths with random parents/fork depths and deduplicated pages."""
    if num_branches < 1:
        raise ValueError("num_branches must be positive")
    if not 0 <= context_remainder < page_size:
        raise ValueError("context_remainder must fit within a page")

    generator = torch.Generator().manual_seed(seed)
    dense_k: list[torch.Tensor] = []
    dense_v: list[torch.Tensor] = []

    for branch_idx in range(num_branches):
        num_full_pages = 1 + branch_idx % 3
        target_len = num_full_pages * page_size + context_remainder
        if branch_idx == 0:
            branch_k, branch_v = _random_tokens(
                target_len,
                num_kv_heads,
                head_dim,
                generator=generator,
                dtype=dtype,
            )
        else:
            parent_idx = int(
                torch.randint(branch_idx, (1,), generator=generator).item()
            )
            parent_k = dense_k[parent_idx]
            parent_v = dense_v[parent_idx]
            max_fork_depth = min(parent_k.shape[0], target_len - 1)
            if branch_idx == 1 and max_fork_depth >= page_size:
                # Guarantee at least one physically shared page while later forks
                # remain random and may happen at non-page-aligned depths.
                fork_depth = page_size
            else:
                fork_depth = int(
                    torch.randint(
                        1,
                        max_fork_depth + 1,
                        (1,),
                        generator=generator,
                    ).item()
                )
            suffix_k, suffix_v = _random_tokens(
                target_len - fork_depth,
                num_kv_heads,
                head_dim,
                generator=generator,
                dtype=dtype,
            )
            branch_k = torch.cat((parent_k[:fork_depth], suffix_k), dim=0)
            branch_v = torch.cat((parent_v[:fork_depth], suffix_v), dim=0)
        dense_k.append(branch_k)
        dense_v.append(branch_v)

    physical_k_pages: list[torch.Tensor] = []
    physical_v_pages: list[torch.Tensor] = []
    page_ids_by_content: dict[tuple[bytes, bytes], int] = {}
    branch_page_ids: list[list[int]] = []

    for branch_k, branch_v in zip(dense_k, dense_v, strict=True):
        page_ids: list[int] = []
        for page_idx in range(ceil(branch_k.shape[0] / page_size)):
            start = page_idx * page_size
            end = min(start + page_size, branch_k.shape[0])
            k_page = torch.full((page_size, num_kv_heads, head_dim), 37.0, dtype=dtype)
            v_page = torch.full_like(k_page, -41.0)
            k_page[: end - start] = branch_k[start:end]
            v_page[: end - start] = branch_v[start:end]
            key = _page_key(k_page, v_page)
            page_id = page_ids_by_content.get(key)
            if page_id is None:
                page_id = len(physical_k_pages)
                page_ids_by_content[key] = page_id
                physical_k_pages.append(k_page)
                physical_v_pages.append(v_page)
            page_ids.append(page_id)
        branch_page_ids.append(page_ids)

    max_pages = max(len(page_ids) for page_ids in branch_page_ids) + 2
    block_tables = torch.full((num_branches, max_pages), -1, dtype=torch.int32)
    for branch_idx, page_ids in enumerate(branch_page_ids):
        block_tables[branch_idx, : len(page_ids)] = torch.tensor(
            page_ids, dtype=torch.int32
        )

    num_q_heads = num_kv_heads * gqa_ratio
    q = torch.randn(
        (num_branches, num_q_heads, head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)
    return TreeAttentionCase(
        q=q,
        k_cache=torch.stack(physical_k_pages),
        v_cache=torch.stack(physical_v_pages),
        block_tables=block_tables,
        context_lens=torch.tensor(
            [branch.shape[0] for branch in dense_k], dtype=torch.int32
        ),
        dense_k=tuple(dense_k),
        dense_v=tuple(dense_v),
    )


def dense_sdpa_oracle(
    case: TreeAttentionCase, *, scale: float | None = None
) -> torch.Tensor:
    """Evaluate every materialized branch independently with PyTorch SDPA."""
    outputs: list[torch.Tensor] = []
    num_q_heads = case.q.shape[1]
    num_kv_heads = case.k_cache.shape[2]
    repeats = num_q_heads // num_kv_heads

    for branch_idx, (branch_k, branch_v) in enumerate(
        zip(case.dense_k, case.dense_v, strict=True)
    ):
        query = case.q[branch_idx].float().unsqueeze(0).unsqueeze(2)
        key = branch_k.float().permute(1, 0, 2).unsqueeze(0)
        value = branch_v.float().permute(1, 0, 2).unsqueeze(0)
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        outputs.append(output.squeeze(0).squeeze(1))
    return torch.stack(outputs).to(case.q.dtype)
