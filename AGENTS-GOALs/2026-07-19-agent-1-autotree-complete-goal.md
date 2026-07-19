# AutoTree — project complete on this machine (2026-07-19)

## State at handoff
Main at `58ad8c7`. Every machine-buildable phase of CLAUDE.md is built, gated, and merged.
All lanes closed, all worktrees removed, fleet ledger updated.

## What landed since wave 1
- Full CPU implementation of all six packages, all suites green (~340 tests), real-engine
  serve gate wired into verify-local + CI (builds the scheduler wheel, un-skips e2e).
- Scheduler-KV lifecycle desync (merge pruned branches without informing the Rust
  scheduler) root-caused and fixed; pinned by the previously-failing 7 serve e2e tests.
- **A100 GPU validation** (Lambda, ~$1.60): Triton kernel parity 106 passed; Qwen3-8B
  bf16 modeling 13 passed; full core on GPU 213 passed. Three GPU-only bugs found and
  fixed (cuda vs cuda:0 canonicalization; device-blind tests; cross-kernel bitwise
  asserts replaced by scoped contract). Evidence + measured kernel benchmark (31.6x at
  32 branches) in `core/docs/a100-validation.md`; parity contract in
  `core/docs/tree-kv-spec.md`.
- **Paper + figures** (Phase 4): arXiv-style LaTeX draft compiles under tectonic
  (build: `cd figures && uv run python make_figures.py --out out/ && cd ../paper &&
  tectonic main.tex`); 7-figure pipeline, 19 figure tests; every fixture number labeled;
  real A100 subsection in evaluation.
- **Enterprise packaging** (Phase 6): deploy/helm (lint clean, kubeconform 5/5),
  deploy/k8s render, deploy/slurm, grafana pack (all metric names verified against
  serve/autotree_serve/metrics.py), auth/quota/audit middleware in
  serve/autotree_serve/enterprise.py (serve: 58 tests), docs/enterprise/* with honest
  implemented-vs-roadmap split.

## GPU environment gotchas (hard-won)
- torch 2.12+ ships CUDA-13 wheels only; CUDA 12.8 drivers need
  `torch==2.11.0+cu128` from the cu128 index or CUDA is silently unavailable.
- `uv run` re-syncs to uv.lock and clobbers a hand-pinned torch — use
  `.venv/bin/python -m pytest` directly on GPU boxes.

## Remaining (founder-blocked)
- AUTOTREE-001: GitHub remote + ThoughtBench domain → push, CI live, leaderboard publish.
- AUTOTREE-002 (partially cleared): single-GPU validation done via Lambda; the
  end-to-end 3-10x ThoughtBench numbers still need multi-GPU serving runs
  (8x for 70B-class) — Lambda 8x A100 was $15.92/hr when checked.
