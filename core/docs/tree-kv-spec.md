# Tree-KV Interface Spec v0

Status: NORMATIVE for wave-1 lanes `core-kv` and `tree-attention`.
Owner: orchestrator. Lanes may not edit this file; propose changes in your
lane report and the orchestrator folds them in.

## Constants

- `PAGE_SIZE`: tokens per KV page. Default **16**, configurable at pool init.
- Dtypes: fp32/fp16/bf16 supported end-to-end; tests may use fp32 on CPU.

## KV pool (owned by lane core-kv)

Per layer, K and V are paged tensors shaped
`[num_pages, PAGE_SIZE, num_kv_heads, head_dim]` (device-agnostic; CPU in
tests). The pool provides:

- `alloc_page() -> int` and refcounted `free(page_id)`.
- Copy-on-write: appending/writing to a page whose refcount > 1 first copies
  it for the writing branch, then writes (classic vLLM COW).
- Content-addressed dedup for FULL (frozen) pages: stable hash of raw page
  bytes; `dedup_scan()` re-points identical pages to one physical page and
  frees the duplicates.

## Tree state (owned by lane core-kv)

- A branch is a node in the tree with: `branch_id`, `parent_id`,
  `num_tokens` (its total root-to-node context length), and a
  `block_table: list[int]` of physical page ids covering, in order, the
  root-to-node token path (last page may be partially filled).
- `fork(branch_id) -> new_branch_id`: shares ALL covered pages (refcount++).
  O(1) pages copied at fork time; divergence happens lazily via COW.
- `prune(branch_id)`: refcount-- on covered pages; pages hitting refcount 0
  are freed immediately (instant reclaim). Pruning a node with live children
  is illegal and must raise.
- Exact memory accounting must be observable: `used_pages`,
  `logical_tokens`, `physical_tokens`, and
  `kv_reuse_ratio = logical_tokens / physical_tokens`.

## Kernel contract (owned by lane tree-attention)

Decode-step attention over paged tree KV:

```
tree_attention_decode(
    q,             # [num_branches, num_q_heads, head_dim] - 1 query token per active branch
    k_cache,       # [num_pages, PAGE_SIZE, num_kv_heads, head_dim]
    v_cache,       # same shape as k_cache
    block_tables,  # int32 [num_branches, max_pages], padded with -1
    context_lens,  # int32 [num_branches] - root-to-node token counts
    scale=None,    # defaults to 1/sqrt(head_dim)
) -> out           # [num_branches, num_q_heads, head_dim]
```

- Branch `i` attends to exactly its first `context_lens[i]` tokens as laid
  out by `block_tables[i]` - its root-to-node path and nothing else. This IS
  the branch-aware causal mask for decode.
- GQA: `num_q_heads` is a multiple of `num_kv_heads`.
- `reference_tree_attention_decode` in pure PyTorch is the NORMATIVE
  implementation; it must run on CPU with fp32 accumulation. The Triton
  kernel must match it within rtol=1e-3/atol=1e-3 (fp32) and
  rtol=2e-2/atol=2e-2 (fp16/bf16).
- Prefill in v0 is handled by the reference path (per-branch SDPA over the
  branch's path with standard causal masking inside the new span); a fused
  prefill kernel is Phase 1.5, not wave 1.

## Branch scheduler engine contract (extracted from `scheduler/` v0.1, review-verified)

- `BranchId` is u64; root is 0 and begins Active; IDs never reused; scheduler
  construction emits no initial command (the engine bootstraps root decoding).
- Engine -> scheduler events: `TokenSampled{branch,token,logprob}` (one token,
  one total-budget unit; branch Active, logprob finite),
  `BranchExhausted{branch}` (Active leaf -> Finalized),
  `ValueScored{branch,score}` (external scorer mode only, after that branch's
  TokenSampled; branch live, score finite; a branch awaiting a score cannot
  accept another TokenSampled).
- Scheduler -> engine commands: `ForkAt{branch,width}` (fork Active branch into
  width children, parent -> Expanded), `Continue{branch}` (authorizes exactly
  one decode step), `Kill{branch,reason}` (terminalize + immediate reclamation),
  `Finalize{branch}`.
- `KillReason`: beam_pruned | speculative_kill | branch_budget_exhausted |
  tree_budget_exhausted.
- Python event/command encoding: snake_case `type` field as in the Rust
  crate's PyO3 layer (`token_sampled`, `fork_at`, ...).

## Environment reality (both lanes)

- The dev box is Windows, CPU-only. Triton does not import on Windows: gate
  all triton imports (`importlib.util.find_spec("triton")`) and skip-mark
  GPU/Triton tests. The CPU reference suites are the wave-1 gate.
- Use `uv` for environments:
  `cd core && uv venv --python 3.12 && uv pip install -e ".[dev]"`,
  then `uv run pytest <tests> -q`. torch installs CPU wheels by default here.
