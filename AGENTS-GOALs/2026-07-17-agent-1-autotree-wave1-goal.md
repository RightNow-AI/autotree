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

## STATUS 2026-07-18 00:45 - paused on quota exhaustion (AUTOTREE-003)

Transport changed: the codex plugin companion's write mode is broken
machine-wide (elevated Windows sandbox hangs headless; unelevated breaks
apply_patch; `~/.codex/config.toml` `[windows] sandbox` was switched to
"unelevated" as the documented fallback and codex-cli is back on 0.144.4).
Founder supplied the working method, now the law for lanes:
`codex exec --cd <worktree> --sandbox danger-full-access --model gpt-5.6-sol
"$(cat brief-file)" < /dev/null > lane.log 2>&1`, background; brief always a
file; waves of TWO; log-is-truth watchers (finish marker "tokens used",
crash signature -1073741502, 12-min frozen log). Wrapper processes get killed
on this machine but codex children survive - trust logs, not process state.
Two host crash storms (0xC0000142 = desktop-heap/memory exhaustion; watch
RunPipe-voice/RunPipe-reviewfix Next dev servers at 3-4GB each) interrupted
lanes; recovery = `codex exec ... resume <session-id> "<audit-first prompt>"`.

Per-lane state (orchestrator independently re-gated fresh, all green):

| Lane | Commits | Verified by orchestrator | Remaining | Resume session id |
|---|---|---|---|---|
| scheduler-rs | `4c1bc38` DONE (21 files, 3,492 lines) | cargo fmt+clippy+test all pass | nothing - ready for merge review | 019f71a3-f16f-7f02-876f-cdceaceab0b5 |
| core-kv | `1ea8c5d`, `bfe833b`, `9ed2d9a` | `71 passed` in tests/kv | commit property test, COW+prune red-run mutation proofs, __init__ exports, core/README.md, final clean gate | 019f71a3-b6df-76e0-afe1-8ca13e805e32 |
| tree-attention | `bf0d391` | `62 passed, 45 skipped` in tests/kernels | commit dispatch.py/bench_decode.py/kernels __init__ + modified reference/tests, final gates; GPU parity stays skip-marked | 019f71d8-7816-7013-9487-372c45eb2be4 |
| serve-api | none - never started | - | full brief from scratch (`scratchpad/brief-serve-api.md`, copy in this repo's history) | none - fresh launch |

Lane logs (session scratchpad):
`C:/Users/jaber/AppData/Local/Temp/claude/C--Users-jaber-RightNow-Full-AutoTree/be156b72-bf28-4d23-b997-33e79fd6a35b/scratchpad/lane-*.log`
Resume prompt templates: `resume-core-kv-r3.md` (pattern: audit-first, re-verify
gates fresh, commit green work early, stop honestly on host failure).

Known token spend visible in finish markers: 191,479 + 371,485 (first
attempts) + 295,876 (core-kv r3) + resumed-run totals in the r2 logs.

## MERGED 2026-07-18 ~15:10 (founder word)

Merge train on local `main`: `d7b6291` (core-kv) -> `8131125` (tree-attention)
-> `f0406b2` (scheduler-rs). Zero file overlaps, zero conflicts. Post-merge
gates run by orchestrator on main: Python `135 passed, 45 skipped` (kv +
kernels, fresh uv env); cargo fmt --check + clippy -D warnings clean, 29
tests 0 failed. Pre-merge mutation re-proof on tree-attention: off-by-one
context_len -> `45 failed`, restored -> `62 passed, 45 skipped`. Merged
worktrees removed; branches retained. No push (AUTOTREE-001: no remote yet).
Remaining: serve-api lane building in `../AutoTree-serve-api` (first commit
`9e77b2b` exists); on completion: orchestrator gate, merge on founder word,
then wave 2 (correctness harness vs real 8B on GPU box, ThoughtBench).

## WAVE 2 COMPLETE 2026-07-18 ~17:50 - awaiting founder merge word

Five branches done + orchestrator-gated (fresh evidence, my own runs):
- `feat/serve-api` `dd7703c`: 26/26; four review P2s fixed red-first and
  re-verified line-by-line (modern OpenAI fields, include_usage, strict
  token_index, branch-relationship validation). MERGEABLE.
- `feat/model-executor` (5 commits to `d635e4e`): GPT-2 executes through
  PagedKVPool/TreeState on CPU; parity harness (bit-parity tree-vs-seq,
  greedy equality vs transformers, KV-reuse evidence, mutation pins);
  17/17 modeling + full core 152 passed/45 skipped. GPU/8B honestly
  unclaimed - harness parameterized for the GPU box.
- `fix/scheduler-hardening` (6 commits to `7091488`): review's P1+3xP2+P3
  fixed red-first (policy-command validation, pinned portable PRNG + golden
  stream, transactional PyO3 polling, continuation reservation, signed
  zero); 35/35, fmt/clippy/python-feature clean.
- `feat/rollout-sdk` (3 commits to `b3a3ba2`): typed tree client + rollout()
  + GRPO/RLHF exports; 18/18. Open item: `final_scores` wire association
  underspecified - reconcile in spec at integration.
- `chore/ci` (2 commits to `5a74346`): workflows (ci + gpu-parity dispatch
  skeleton) + verify-local.ps1/.sh; my end-to-end run: all PASS, serve/
  modeling honest SKIPs.

Transport law (hard-won): WMI-detached codex exec via Foundry;
<=4 concurrent sol streams MACHINE-WIDE (shared deployment rate limits -
raise TPM in Foundry portal to lift); takeover-brief recovery on death;
commit early.

Next: founder merge word -> merge train (serve-api, model-executor,
scheduler-hardening, rollout-sdk, ci) + post-merge full gates; core-review
rerun; then engine-integration lane (scheduler PyO3 + ModelExecutor + serve
EngineProtocol = real `--engine treekv`); then GPU box (AUTOTREE-002) for
parity at 8B and Triton validation; remote (AUTOTREE-001) lights up CI.

## WAVE 2 MERGED 2026-07-18 ~18:05: main `5397a2e` (5-branch train, zero conflicts; post-merge verify-local incl modeling ALL PASS + sdk 18/18). Wave 3 launched: engine-integration lane running (feat/treekv-engine, owns core engine + serve wiring + final_scores fix); thoughtbench lane queued on capacity (feat/thoughtbench, fixture-only harness). Then: core-review rerun, GPU box (AUTOTREE-002) for 8B parity + Triton, remote (AUTOTREE-001) for CI.

## CPU-COMPLETE 2026-07-19 ~02:00: main `2d592f1` carries the full blueprint CPU surface: Tree-KV engine, kernels (reference+Triton-unvalidated), hardened scheduler, OpenAI-compatible serving w/ live playground, SDK w/ contract-tested wire, thoughtbench, leaderboard, docs+Dockerfile, 7-figure pipeline, CI. Audit Wave A done except serve-ops lane (running). Next: GPU box (AUTOTREE-002) unlocks parity@8B + Triton + real numbers (audit Wave B: forest batching, SGLang TreeRadix); remote (AUTOTREE-001) lights CI.

## Next session

Read this file, check the four job ids, gate each lane's diff (companion
adversarial-review for the kernel and the EngineProtocol seam), fold
reported spec changes into `tree-kv-spec.md`, then merge train on founder
word. After wave 1: correctness-harness lane (bit-parity vs sequential on a
real 8B on the GPU box), then ThoughtBench (Phase 3).

## WIRE CONTRACT COMPLETE 2026-07-19 00:30 +03:00

Lane `fix/wire-contract` completed locally with no push and no configured
remote/upstream:

- `df878c7` pinned the real serve-to-SDK stream contract, fork-prefix exports,
  and the core `kv_reuse_ratio >= 1` metric behavior red-first.
- `ff9274e` added finite sampled-token `logprob` and prune `reason` fields to
  deterministic and TreeKV event emission and server validation.
- `04072ea` made GRPO and RLHF exports reconstruct complete root-to-leaf token
  paths and made the SDK parser enforce the full branch-keyed summary contract.
- `c5f1c3a` moved ThoughtBench to `thoughtbench.results.v2`, accepts the core
  logical/physical KV multiplier, and reads it from the typed tree summary.
- `6c39351` added the normative `core/docs/wire-spec.md` v1 contract.
- `e8e242e` added separate SDK, real cross-package wire, and ThoughtBench CI
  jobs plus matching PowerShell and shell local-verification gates.
- `3940a6d` closed a manual-review gap by reconciling prune/merge counts and
  rejecting invalid counters and non-finite final scores server-side.

Verification on the final implementation:

- `scripts/verify-local.ps1`: PASS for core (`135 passed, 45 skipped`), all
  scheduler fmt/clippy/test/python-feature gates, serve (`30 passed, 3 skipped`
  without the optional scheduler wheel), SDK unit (`19 passed`), real
  serve-to-SDK contract (`1 passed`), ThoughtBench (`34 passed`), and workflow
  YAML parsing.
- Scheduler release wheel built with maturin and installed into the serve env;
  the complete serve suite then passed `33 passed`, including `3 passed` real
  GPT-2 TreeKV tests. The complete SDK suite passed `20 passed` and ThoughtBench
  passed `34 passed`.
- Captured raw TreeKV SSE contained finite token logprobs
  `-1.0208925008773804`, `-0.5032301545143127`,
  `-0.5032301545143127`, and `-0.29521721601486206`; prune reasons were present;
  done usage reported 4 completion tokens for 4 token events; counters were
  logical 33 / physical 12 and `kv_reuse_ratio` was 2.75.
- External CodeRabbit review was unavailable because the CLI is not installed.
  Manual full-range review found the terminal-summary validation gap above;
  its pins failed `2 failed, 7 passed` before the fix and passed `9 passed`
  afterward.

Remaining gaps: the optional modeling gate was not requested and remained
skipped in verify-local; no push, remote sync, deployment, or GPU claim was made.

## ENGINE SEMANTICS REGRESSION FIX 2026-07-19 18:16 +03:00

Lane `fix/engine-semantics` commit `72590ce` fixes the real-engine TreeKV
lifecycle regression without changing default KV capacity or weakening the
banked logprob, mean-ranking, EOS, SDK-alternative, or stop-scoring semantics.
The engine now reconciles convergence with scheduler state, excludes expanded
branches from merge reclamation, drains commands after batch reconciliation,
and drops stale continuations before decode. The scheduler Python binding
exposes read-only branch state so engine and scheduler topology stay aligned.

Regression proof: the new real-scheduler dedup lifecycle test failed before the
fix with `scheduler stopped issuing commands with active branches remaining`
and passes for beam, best-first, and MCTS after the fix. The seven reported
serve reproductions pass together (`7 passed`). Final mandatory gates after
rebuilding/installing the release scheduler wheel and `core[engine]`: serve
`50 passed` with zero TreeKV skips, core `171 passed, 45 skipped`, SDK
`21 passed`. Scheduler `cargo fmt --check`, clippy with warnings denied, and
default cargo tests passed; direct Windows `cargo test --features python`
still hits host `STATUS_DLL_NOT_FOUND`, while the release wheel import/state
smoke test and full real-engine serve suite pass.

Review: CodeRabbit CLI was unavailable, so no external review ran. Manual
full-diff review found no remaining critical or warning issue. No push, remote
sync, deployment, or GPU claim was made.
