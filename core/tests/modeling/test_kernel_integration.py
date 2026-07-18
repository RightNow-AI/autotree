from __future__ import annotations

import torch

from autotree_core.kernels import reference_tree_attention_decode

from .conftest import ModelCase


def test_real_decode_model_attention_matches_reference_tree_kernel(
    model_case: ModelCase,
) -> None:
    executor = model_case.executor
    execution = executor.prefill(model_case.prompt[:5])
    attention = executor.model.transformer.h[0].attn
    captured: dict[str, torch.Tensor] = {}

    def capture_hidden(
        _module: torch.nn.Module, args: tuple[torch.Tensor, ...]
    ) -> None:
        captured["hidden"] = args[0].detach().clone()

    def capture_attention_output(
        _module: torch.nn.Module, args: tuple[torch.Tensor, ...]
    ) -> None:
        captured["attention"] = args[0].detach().clone()

    hidden_handle = attention.register_forward_pre_hook(capture_hidden)
    output_handle = attention.c_proj.register_forward_pre_hook(capture_attention_output)
    try:
        token_id = int(torch.argmax(execution.next_logits(execution.root_id)).item())
        executor.decode(execution, execution.root_id, token_id)
    finally:
        hidden_handle.remove()
        output_handle.remove()

    query_projection = attention.c_attn(captured["hidden"]).split(
        attention.split_size, dim=2
    )[0]
    query = query_projection.view(
        1, 1, executor.num_attention_heads, executor.head_dim
    )[:, 0]
    block_tables, context_lens = execution.attention_metadata([execution.root_id])
    kernel_output = reference_tree_attention_decode(
        query,
        execution.pool.k_cache[0],
        execution.pool.v_cache[0],
        block_tables,
        context_lens,
        scale=float(attention.scaling),
    )
    model_attention = captured["attention"].view(
        1, 1, executor.num_attention_heads, executor.head_dim
    )[:, 0]

    torch.testing.assert_close(
        kernel_output,
        model_attention,
        rtol=1e-6,
        atol=1e-6,
    )
