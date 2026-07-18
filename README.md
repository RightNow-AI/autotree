# AutoTree

<!-- markdownlint-disable MD013 -->

AutoTree is an experimental engine for executing LLM reasoning as a tree. It
shares prefix KV state, forks candidate branches, and lets a Rust scheduler
prune or continue them under a token budget.

Today, this repository provides a CPU demonstration with real GPT-2 weights,
the Tree-KV data structures and reference kernels, an OpenAI-style HTTP API,
a typed Python SDK, and a fixture-only benchmark harness. GPU kernels,
large-model validation, production serving, and performance claims are future
phases. The blueprint's 3-10x cost reduction and 5x rollout-throughput targets
are hypotheses to be tested, not results from the current code.

## Five-minute CPU quickstart

Prerequisites: Windows PowerShell, [uv](https://docs.astral.sh/uv/), and Rust
1.93 or newer. Start in the repository root after cloning it. The first server
start downloads GPT-2 from Hugging Face, so network speed determines the total
time.

Build the Rust scheduler wheel and install the CPU engine and server:

```powershell
uv venv --python 3.12 --clear
uvx maturin build --release --features python --manifest-path scheduler/Cargo.toml --out dist
$wheel = Get-ChildItem dist/autotree_scheduler-*.whl | Select-Object -First 1
uv pip install -e './core[engine]' -e ./serve $wheel.FullName
```

Start the real CPU TreeKV demo:

```powershell
uv run --no-project autotree serve --engine treekv --model gpt2
```

When Uvicorn reports that it is running on `http://127.0.0.1:8000`, open a
second PowerShell window in the repository root and request a tree completion:

```powershell
@'
{"model":"gpt2","messages":[{"role":"user","content":"Explain why shared prefixes matter."}],"max_tokens":4,"seed":7,"tree":{"policy":"beam","branches":3,"budget_tokens":12}}
'@ | curl.exe --silent --show-error --fail http://127.0.0.1:8000/v1/tree/completions --header "Content-Type: application/json" --data-binary '@-'
```

The response is an HTTP 200 chat-completion object with generated text,
token usage, and a `tree` summary containing branch outcomes, scores, token
spend, and KV reuse. See [the full quickstart](docs/quickstart.md) for the
captured output and troubleshooting.

With the server running, the live tree playground is at
`http://127.0.0.1:8000/playground`: paste a prompt and watch branches fork,
prune, and merge in real time. It is a fully offline page served by
`autotree-serve` itself.

## Packages

| Path | Current responsibility |
| --- | --- |
| `core/autotree_core/kv/` | Device-agnostic paged KV pool, copy-on-write forks, pruning, deduplication, gathers, and accounting |
| `core/autotree_core/kernels/` | Normative PyTorch tree-attention reference plus an import-guarded Triton decode path |
| `core/autotree_core/engine/` | CPU model executor and TreeKV engine that connect Hugging Face weights, KV state, and scheduling |
| `scheduler/` | Deterministic Rust beam, best-first, and MCTS policies with token-budget enforcement and optional PyO3 bindings |
| `serve/` | `autotree` CLI, OpenAI-style chat/tree endpoints, SSE streaming, and Prometheus metrics |
| `sdk/` | Typed Python client and rollout-trace exports for GRPO/RLHF-shaped consumers |
| `thoughtbench/` | Fixture-only benchmark and report harness; it is not evidence for accuracy, cost, or performance claims |

The normative current Tree-KV contract is
[`core/docs/tree-kv-spec.md`](core/docs/tree-kv-spec.md). For how the pieces
fit together, see [docs/architecture.md](docs/architecture.md).

## Development and verification

Run the repository's CPU-local verification from PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/verify-local.ps1
```

Add `-Modeling` to that command when you intentionally want the slower model
download and modeling suite. GPU/Triton parity is not established by the
local Windows gate.

## Docker

The root `Dockerfile` builds the Rust scheduler wheel in a Python 3.12 builder
and produces a non-root CPU image whose default command serves GPT-2 through
TreeKV on port 8000. On a machine with Docker, build it with
`docker build -t autotree-cpu .`. Docker was not available on the Windows box
used to verify this documentation, so the image recipe was linted but not
built there.

## Status and roadmap

The complete target architecture and phase gates live in
[`CLAUDE.md`](CLAUDE.md). Treat it as a roadmap, not a description of already
shipped behavior. Current limitations and the implemented OpenAI-compatible
surface are summarized in [docs/faq.md](docs/faq.md).
