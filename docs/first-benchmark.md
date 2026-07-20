# First real ThoughtBench measurement — 2026-07-20

Single H100 PCIe 80GB (Lambda), Qwen/Qwen3-8B bfloat16 served by the reference
engine (`autotree serve --engine treekv --device cuda --dtype bfloat16`).
Tasks: 25 numeric MATH-500 problems (levels 1-3), real provenance, 3 seeds.
Result files: `thoughtbench/results/` and `leaderboard/results/` (schema-validated,
`benchmark_claims_allowed: true`).

| Arm | acc@1 | acc@4 | tokens/correct | KV-reuse |
|---|---|---|---|---|
| Sequential best-of-4, 640 tok/sample | 30.7% | 56.0% | 3,159 | n/a |
| Tree beam-8, 1,920 budget tokens | 21.3% | - | 5,011 | **8.79x** |

## What this establishes

- **Mechanism confirmed end-to-end**: 8.79x KV reuse on real workloads - branches
  share prefix KV exactly as the kernel-level A100 result predicted.
- **Honest negative on naive quality**: log-probability winner selection with
  ~240 tokens/branch loses to independent sampling on solvable tasks. The
  pluggable value scorer is the component this baseline exists to motivate;
  any value-guided configuration must beat these numbers.
- **AIME infeasibility documented**: at reference-engine speed (~3.4 tok/s/stream)
  AIME-scale thinking budgets truncate every sample (0/5 partial evidence
  retained). Frontier-difficulty rows require production-rate serving.

Protocol, configs, and full sample-level data ship in the results JSONs.
GPU cost of this run: ~USD 28 (8.5 h x USD 3.29/h).
