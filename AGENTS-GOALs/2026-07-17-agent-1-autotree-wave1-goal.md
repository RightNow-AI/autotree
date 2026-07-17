# AutoTree wave 1 - Phase-1 engine + Phase-2 prep lanes

Date: 2026-07-17. Orchestrator: Fable (ultrateam). Repo initialized this
session at `5a5b02c` on `main` (local only, no remote yet).

## Mission

Start executing the CLAUDE.md blueprint: Phase 1 Tree-KV engine (COW paged
KV, fork/merge/prune, tree attention) plus the Rust scheduler and the
OpenAI-compatible API layer, as four isolated Codex gpt-5.6-sol lanes.

## Lanes (launched via the ultrateam companion launcher)

| Lane | Branch | Worktree | Job id | Scope |
|---|---|---|---|---|
| core-kv | feat/core-kv | `../AutoTree-core-kv` | task-mrp7i8oc-e19vwf | `core/autotree_core/kv/**` + tests: COW paged pool, fork/prune, content-addressed dedup, exact accounting |
| tree-attention | feat/tree-attention | `../AutoTree-tree-attention` | task-mrp7icki-tit4kt | `core/autotree_core/kernels/**` + tests: normative torch reference + Triton decode kernel (import-guarded) |
| scheduler-rs | feat/scheduler-rs | `../AutoTree-scheduler-rs` | task-mrp7ifxp-77zt84 | `scheduler/**`: beam/best-first/MCTS, ValueScorer, token budgets, speculative kill, PyO3 behind `python` feature |
| serve-api | feat/serve-api | `../AutoTree-serve-api` | task-mrp7ijph-mnclsf | `serve/**`: OpenAI-compatible API, /v1/tree/completions, per-branch SSE events, Prometheus, CLI, EngineProtocol seam + DeterministicEngine |

`status`/`result` must use the SAME `--lane` path each task was launched
from (job state is keyed by launch CWD).

## Shared contract

`core/docs/tree-kv-spec.md` is NORMATIVE for the two core lanes and
orchestrator-owned, as is `core/pyproject.toml`. Lanes request changes via
their reports; the orchestrator folds them in.

## Gates (per lane, evidence required)

CPU pytest green (Python lanes) / fmt + clippy -D warnings + test green
(Rust lane); pins with red-run proofs; zero conflict markers; commits local
only - no pushes. Merge train on founder word: per-branch rebase onto main,
re-gate, merge, report SHA.

## Environment facts (verified this session)

- Windows CPU-only box; bare `python` on PATH is the broken Store stub - use
  uv 0.9.28 with `uv venv --python 3.12`. Rust 1.93 installed. Triton does
  not import on Windows, so GPU/Triton tests are skip-marked by design.
- Codex transport healthy: plugin 1.0.6, codex-cli 0.144.4, API-key auth.

## Blockers (also filed in RunPipe/ultramerge/BLOCKERS.md)

- AUTOTREE-001: founder to reserve the ThoughtBench domain + GitHub org and
  create the AutoTree remote (repo is local-only; CI and pushes blocked).
- AUTOTREE-002: Phase-1 parity gate (greedy bit-parity vs sequential on
  Qwen3-8B / Llama-3.1-8B) and Triton kernel validation need a Linux GPU box
  (8xH100 target).

## Next session

Read this file, check the four job ids, gate each lane's diff (companion
adversarial-review for the kernel and the EngineProtocol seam), fold
reported spec changes into `tree-kv-spec.md`, then merge train on founder
word. After wave 1: correctness-harness lane (bit-parity vs sequential on a
real 8B on the GPU box), then ThoughtBench (Phase 3).
