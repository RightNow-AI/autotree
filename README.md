# AutoTree

> vLLM owns chat. AutoTree owns thought.

Kernel-level tree execution for LLM test-time compute: fork, merge, and prune
KV cache at token granularity so deep reasoning costs 3-10x less and RL
rollout generation runs >=5x faster, at identical quality.

Full build blueprint: [CLAUDE.md](./CLAUDE.md). License: Apache-2.0.

## Monorepo layout

| Path | Package | What |
|---|---|---|
| `core/` | `autotree-core` (Python) | Tree-KV engine: COW paged KV pool, fork/merge/prune, content-addressed dedup (`autotree_core.kv`) plus tree-attention reference and Triton kernels (`autotree_core.kernels`) |
| `scheduler/` | `autotree-scheduler` (Rust) | Branch policy engine: beam / best-first / MCTS, value-guided pruning, token budgets; PyO3 bindings |
| `serve/` | `autotree-serve` (Python) | OpenAI-compatible server, `/v1/tree/completions` extension, CLI |
| `core/docs/tree-kv-spec.md` | - | Normative Tree-KV interface contract shared across packages |

## Dev

Python packages use [uv](https://docs.astral.sh/uv/):

```
cd core && uv venv --python 3.12 && uv pip install -e ".[dev]" && uv run pytest
```

Rust:

```
cd scheduler && cargo test
```

This dev box is CPU-only; all CPU suites must pass here. GPU/Triton tests are
skip-marked and run on the Linux GPU box.
