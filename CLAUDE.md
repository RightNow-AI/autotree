# AutoTree — Build Blueprint & E2E Prompt

> Positioning: **vLLM owns chat. AutoTree owns thought.**
> Kernel-level tree execution for test-time compute: fork, merge, and prune KV cache at token granularity so deep reasoning costs 3–10× less and RL rollout generation runs ≥5× faster — at identical quality.

---

## 1. What AutoTree is

Reasoning models scale with *thinking tokens*. But today's serving engines (vLLM, SGLang, TRT-LLM) execute every sampled chain as if it were alone — recomputing prefixes that sibling branches share, and burning GPU-hours on branches a cheap value signal could have killed at token 50.

AutoTree executes reasoning **as a tree**:

- **Fork** a KV cache at any token with copy-on-write (COW) — near-zero cost to spawn a branch.
- **Merge** sibling branches that converge, deduplicating KV blocks content-addressably.
- **Prune** low-value branches early with a pluggable scorer, reclaiming memory instantly.
- Expose all of it through a **drop-in OpenAI-compatible API**, so adoption = changing one `base_url`.

Two customer pools:

1. **Everyone running reasoning models** (viral, open-source, OpenClaw-style spread) — free, self-hosted, instant savings.
2. **Frontier labs doing RL post-training** (the whales) — rollout generation is their dominant post-training cost; AutoTree multiplies throughput on the same GPUs.

---

## 2. Tech stack (fixed — do not deviate)

| Layer | Choice |
|---|---|
| Kernels | Triton (primary) + CUDA C++ (hot paths), branch-aware tree-attention masking, FlashInfer-style primitives, NVFP4 paths for Blackwell |
| KV memory | Paged KV allocator with COW fork/merge, content-addressed block dedup, GPU-direct RDMA transfer (NCCL / NVSHMEM) for multi-node |
| Scheduler core | **Rust** (latency-critical branch policy engine), PyO3 bindings |
| Serving | Fork of **SGLang** (RadixAttention is the closest existing substrate to tree-KV) → `autotree-serve`; vLLM plugin as stretch goal |
| API | OpenAI-compatible REST + streaming, plus `/v1/tree/completions` extension; Python SDK + CLI |
| Models (ladder) | Llama-3.1-8B / Qwen3-8B (dev) → Llama-3.3-70B / Qwen3-32B / R1-Distill-70B (MVP bench) → **Inkling-Small 276B** (8×B200, NVFP4) → **Inkling 975B** (2–4 nodes, NVFP4) |
| Orchestration | Kubernetes + Helm chart + K8s operator; SLURM scripts for HPC/labs |
| Observability | Prometheus + Grafana + OpenTelemetry (per-branch traces) |
| Bench harness | Python, automated graders, 3 seeds, published configs, one-command repro |
| Leaderboard | Next.js static site ("ThoughtBench"), downloadable JSON |
| Paper figures | Matplotlib/seaborn, publication style, PDF/SVG 300dpi |
| License | Apache-2.0 |

---

## 3. System architecture

```
┌────────────────────────── AutoTree ──────────────────────────┐
│ SDK / CLI / OpenAI-compatible API  (/v1/tree/completions)     │
├──────────────────────────────────────────────────────────────┤
│ Rust Branch Scheduler                                         │
│  policies: beam | best-first | MCTS                           │
│  value-head-guided pruning | token-budget controller          │
│  speculative branch kill | Inkling effort-knob integration    │
├──────────────────────────────────────────────────────────────┤
│ Tree-KV Engine                                                │
│  COW paged allocator · fork/merge · block dedup               │
│  Triton tree-attention kernel · CUDA hot paths · NVFP4        │
├──────────────────────────────────────────────────────────────┤
│ Scale layer: GPU-direct RDMA KV transfer · P/D disaggregation │
├──────────────────────────────────────────────────────────────┤
│ Observability: Prometheus · Grafana · OTel branch traces      │
└──────────────────────────────────────────────────────────────┘
```

**Component specs:**

1. **Tree-KV Engine** — the moat. Token-granularity branch points; COW so a fork costs O(1) pages; merge via block-hash dedup; tree-attention kernel with branch-aware causal masks (a token attends only to its root-to-node path).
2. **Branch Scheduler** — decides *where* to branch, *how wide*, *what to kill*. Ships with beam, best-first, MCTS; accepts a pluggable value scorer (small model or reward head); enforces hard token budgets; maps Inkling's `thinking_effort` parameter to tree width/depth.
3. **Serving** — `autotree-serve`: OpenAI-compatible; tree params via `extra_body={"tree": {...}}`; streaming emits per-branch events; AutoTree OFF = stock SGLang behavior (zero regression risk for adopters).
4. **Scale** — prefill/decode disaggregation; KV migration over RDMA so a full tree never blocks on one node; SLURM for lab clusters.
5. **RL rollout API** — `sdk.rollout(prompts, k, policy)` returning branching traces + logprobs, format-compatible with common RLHF/GRPO stacks.

---

## 4. UX — "clear enough to go viral"

- **Install:** `pip install autotree && autotree serve --model Qwen/Qwen3-32B`
- **Adopt:** change `base_url` to `http://localhost:8000/v1`. Nothing else changes.
- **Use the tree:** one line —
  ```python
  client.chat.completions.create(
      model="Qwen3-32B", messages=msgs,
      extra_body={"tree": {"policy": "beam", "branches": 8, "budget_tokens": 4000}})
  ```
- **Playground (web):** paste a problem, watch the reasoning tree grow/prune live, see cost-per-answer vs. your current stack. This is the shareable demo.
- **SDK (labs):** `autotree.rollout(...)` with traces for RL pipelines.
- **Docs:** quickstart in <5 min, migration guides from vLLM/SGLang, Docker one-liner.

---

## 5. Enterprise integration

- **Deploy:** Helm chart + K8s operator (autoscaling on KV pressure), on-prem/VPC/air-gapped installs, SLURM templates, Docker images for x86 H100/B200.
- **Security/Governance:** OIDC/SAML SSO, RBAC, full audit logs, per-tenant quotas; SOC2 roadmap documented from day one.
- **Compat:** vLLM/SGLang flag compatibility mode; HF + S3 model registries; LangChain/LlamaIndex drop-in via OpenAI client; Tinker/RL-stack integration guide for labs.
- **Ops:** prebuilt Grafana dashboards (KV reuse, useful-token ratio, $/correct answer), PagerDuty-ready alerts, SLA metrics endpoint.
- **Pricing shape (later):** open-source engine free; enterprise = support SLA + managed control plane + private ThoughtBench.

---

## 6. ThoughtBench — the benchmark that goes viral

**Headline claim to prove:** *same model, same accuracy, 3–10× cheaper* — and *same budget, materially higher accuracy*.

**Model ladder**
- MVP: Llama-3.3-70B, Qwen3-32B, DeepSeek-R1-Distill-70B (single node, 8×H100)
- Wave 2: **Inkling-Small 276B** on 8×B200 with NVFP4 (first public deep-reasoning cost curve on Thinking Machines' stack — the news hook)
- Wave 3: **Inkling 975B** multi-node NVFP4; GLM/DeepSeek/Kimi flagship open weights as available; proprietary APIs (GPT, Claude, Gemini) as cost reference lines only

**Benchmarks:** AIME 2026, GPQA Diamond, Humanity's Last Exam, LiveCodeBench, SWE-bench Verified, Terminal-Bench (agentic).

**Metrics:** accuracy@k (k ∈ {1,4,16,64}), pass^k, **cost per correct answer**, tokens/sec, TTFT, rollout throughput (rollouts/hr/GPU), KV-reuse ratio, useful-token ratio.

**Baselines:** stock vLLM, stock SGLang, AutoTree-OFF (ablation), sequential best-of-n.

**Protocol:** 3 seeds, fixed decoding configs, published model cards + configs, automated graders, one-command repro repo. No cherry-picking — full results downloadable.

**Viral artifacts**
1. Public **ThoughtBench leaderboard** (accuracy-vs-cost Pareto, AutoTree ON/OFF toggle, per-model pages).
2. The **tree render** — a real AIME problem solved as a living tree, branches colored by value estimate. This image travels.
3. Launch post: *"Inkling thinks 5× cheaper when you let it branch"* + arXiv paper same day.

---
AutoTree — Full Build Plan
vLLM owns chat. AutoTree owns thought.
Kernel-level tree execution for test-time compute: fork, merge, prune KV cache at token granularity → deep reasoning 3–10× cheaper, RL rollouts ≥5× faster, identical quality.
1. What AutoTree is
Reasoning scales with thinking tokens, but vLLM/SGLang/TRT-LLM run every sampled chain alone — recomputing shared prefixes, burning compute on branches a cheap value signal could kill at token 50. AutoTree executes reasoning as a tree:
Fork KV at any token, copy-on-write → near-zero cost per branch
Merge converging siblings, content-addressed block dedup
Prune low-value branches early, instant memory reclaim
Drop-in OpenAI-compatible API → adoption = changing one base_url
Customers: everyone running reasoning models (viral open-source spread) + frontier labs doing RL post-training (the whales — rollout generation is their dominant post-training cost).
2. Tech stack (fixed)
Table
Layer	Choice
Kernels	Triton + CUDA C++ hot paths, branch-aware tree-attention masks, FlashInfer-style primitives, NVFP4 for Blackwell
KV memory	COW paged allocator, block dedup, GPU-direct RDMA (NCCL/NVSHMEM) multi-node
Scheduler	Rust core + PyO3 bindings
Serving	Fork of SGLang (RadixAttention ≈ tree-KV substrate) → autotree-serve; vLLM plugin stretch
API	OpenAI-compatible + /v1/tree/completions; Python SDK + CLI
Model ladder	Qwen3-8B (dev) → Llama-70B / Qwen3-32B / R1-Distill-70B (MVP) → Inkling-Small 276B (8×B200 NVFP4) → Inkling 975B (multi-node)
Orchestration	K8s + Helm + operator; SLURM for labs
Observability	Prometheus + Grafana + OTel per-branch traces
Leaderboard	Next.js static site ("ThoughtBench"), downloadable JSON
Figures	Matplotlib, 300dpi PDF/SVG
License	Apache-2.0
3. Architecture
plain
API / SDK / CLI (/v1/tree/completions)
→ Rust Branch Scheduler (beam | best-first | MCTS, value-guided pruning, budget controller, Inkling effort-knob mapping)
→ Tree-KV Engine (COW fork/merge, dedup, Triton tree-attention, NVFP4)
→ Scale layer (RDMA KV transfer, prefill/decode disaggregation)
→ Observability (Prometheus, Grafana, branch traces)
Plus sdk.rollout(prompts, k, policy) returning traces + logprobs for RL pipelines (the labs' entry point). AutoTree OFF = stock SGLang behavior — zero regression risk for adopters.
4. UX
pip install autotree && autotree serve --model Qwen/Qwen3-32B
Adopt = change base_url. Nothing else.
Use the tree in one line: extra_body={"tree": {"policy":"beam","branches":8,"budget_tokens":4000}}
Web playground: watch a reasoning tree grow/prune live + your $/answer vs. current stack — the shareable demo
Docs: <5 min quickstart, vLLM/SGLang migration guides, Docker one-liner
5. Enterprise integration
Helm + operator (autoscale on KV pressure), on-prem/VPC/air-gapped, SLURM templates · OIDC/SAML SSO, RBAC, audit logs, tenant quotas, SOC2 roadmap · vLLM/SGLang flag-compat mode, HF+S3 registries, LangChain/LlamaIndex drop-in, RL-stack guides · Grafana packs ($/correct answer, KV reuse), SLA endpoint. Pricing later: engine free; enterprise = SLA + managed control plane + private ThoughtBench.
6. ThoughtBench — the viral benchmark
Claim to prove: same model, same accuracy, 3–10× cheaper — or same budget, materially higher accuracy.
Models: MVP 70B-class single node → Inkling-Small 276B on 8×B200 (first public deep-reasoning cost curve on Thinking Machines' stack — the news hook, and the window is open now) → Inkling 975B + GLM/DeepSeek/Kimi flagships; proprietary APIs as reference lines
Benchmarks: AIME 2026, GPQA Diamond, HLE, LiveCodeBench, SWE-bench Verified, Terminal-Bench
Metrics: accuracy@k {1,4,16,64}, pass^k, cost per correct answer, tokens/sec, TTFT, rollouts/hr/GPU, KV-reuse ratio, useful-token ratio
Baselines: stock vLLM, stock SGLang, AutoTree-OFF, sequential best-of-n
Protocol: 3 seeds, fixed configs, automated graders, one-command repro, all data downloadable — no cherry-picking
Artifacts: public leaderboard with ON/OFF toggle · the tree render (real AIME solve, branches colored by value) — that image travels · launch post "Inkling thinks 5× cheaper when you let it branch" + arXiv same day
7. Paper figures (scientific spec)
Accuracy-vs-tokens scaling curves, 4 models, AutoTree vs sequential
Cost-vs-accuracy Pareto per model vs vLLM/SGLang — the money chart
KV-reuse heatmap (tree depth × branching factor)
Tree topology render of a real AIME solve — Figure 1
Branching ablation k ∈ {1,2,4,8,16,32}
Rollout throughput bars vs baselines — the RL-labs slide
Inkling thinking_effort sweep (0.2→0.99) × AutoTree ON/OFF
Appendix: determinism/parity check across batch sizes.
8. Roadmap & gates
Table
Phase	Weeks	Gate
1 — Tree-KV engine	1–4	Bit-parity (greedy) vs sequential on 8B, CI green
2 — Scheduler + serving	5–8	70B on 8×H100 via OpenAI client, base_url swap only
3 — ThoughtBench suite	9–12	≥3× cheaper iso-accuracy AIME k=64; ≥5× rollout throughput; parity within noise
4 — Leaderboard + figures	10–13	Site live, figures regenerate from data, arXiv draft
5 — Inkling scale	12–16	Inkling-Small → Inkling rows on leaderboard
6 — Enterprise packaging	14–18	Fresh-cluster install <30 min from docs
9. The E2E prompt (copy-paste to your builder agent)
plain
You are a world-class GPU systems and infrastructure engineering team. Build AutoTree v0.1:
an open-source tree-execution engine for LLM test-time compute.

CONTEXT
Reasoning models scale with thinking tokens, but current serving engines (vLLM, SGLang)
execute every sampled chain independently, recomputing shared prefixes and wasting compute
on low-value branches. AutoTree executes reasoning as a TREE: fork/merge KV cache at token
granularity with copy-on-write, prune low-value branches with a pluggable value scorer, and
enforce hard token budgets. Goal: identical quality at 3-10x lower cost, and >=5x RL rollout
throughput. Positioning: "vLLM owns chat. AutoTree owns thought."

HARD CONSTRAINTS
- Linux, Python 3.12, PyTorch 2.x, Triton, CUDA 12.x, Rust stable (scheduler core, PyO3 bindings)
- Targets: 8xH100 (dev/MVP), 8xB200 with NVFP4 (scale)
- Serving = fork of SGLang exposed as an OpenAI-compatible server; AutoTree OFF mode must be
  behavior-identical to stock SGLang
- Apache-2.0. Everything reproducible with one command. No placeholders, no TODO stubs.

PHASE 1 — TREE-KV ENGINE (package: autotree-core)
- Paged KV allocator with copy-on-write fork/merge at token granularity and
  content-addressed block dedup
- Tree-attention kernel in Triton (branch-aware causal mask: a token attends only to its
  root-to-node path); CUDA hot paths where Triton is the bottleneck
- Correctness harness: greedy bit-parity vs. sequential execution on Qwen3-8B and
  Llama-3.1-8B; seeded statistical parity for sampling; all in CI
GATE: parity tests pass; unit tests for fork/merge/dedup pass.

PHASE 2 — SCHEDULER + SERVING (package: autotree-serve)
- Rust scheduler: policies beam / best-first / MCTS; pluggable value-head scorer for pruning;
  token-budget controller; speculative branch kill with immediate memory reclamation
- OpenAI-compatible endpoints: /v1/chat/completions, streaming; extension
  /v1/tree/completions accepting {"policy","branches","budget_tokens","scorer"}
- Prometheus metrics: KV-reuse ratio, useful-token ratio, branch counts, tokens/sec, TTFT
- Docker image; `autotree serve --model <hf-id>` one-command startup
GATE: serve Llama-3.3-70B on 8xH100; OpenAI client works by changing base_url only.

PHASE 3 — THOUGHTBENCH BENCHMARK SUITE (package: thoughtbench)
- Benchmarks: AIME 2026, GPQA Diamond, LiveCodeBench
- Models: Llama-3.3-70B, Qwen3-32B, DeepSeek-R1-Distill-70B
- Baselines: stock vLLM, stock SGLang, AutoTree-OFF, sequential best-of-n
- Metrics: accuracy@k for k in {1,4,16,64}; cost per correct answer; tokens/sec; TTFT;
  rollout throughput (rollouts/hr/GPU); KV-reuse ratio; useful-token ratio
- Protocol: 3 seeds, fixed decoding configs, automated graders, results as downloadable JSON,
  one-command repro
GATE: >=3x cost reduction at iso-accuracy on AIME 2026 with k=64 vs. best baseline;
>=5x rollout throughput; accuracy parity within noise.

PHASE 4 — LEADERBOARD + PAPER FIGURES
- Next.js static site "ThoughtBench": accuracy-vs-cost Pareto chart, per-model pages,
  AutoTree ON/OFF toggle, downloadable data
- 7 publication figures (matplotlib, 300dpi PDF+SVG, error bars over 3 seeds):
  (1) accuracy-vs-tokens scaling curves, 4 models, AutoTree vs sequential
  (2) cost-vs-accuracy Pareto per model vs vLLM/SGLang
  (3) KV-reuse heatmap by tree depth x branching factor
  (4) tree topology render of a real AIME solve (color = value estimate, pruned = red)
  (5) branching-factor ablation k in {1,2,4,8,16,32}
  (6) rollout throughput bar chart vs baselines
  (7) thinking-effort sweep x AutoTree ON/OFF (prepare now; fill with Inkling in Phase 5)
- arXiv-style paper draft (LaTeX) with the figures integrated
GATE: site renders from results JSON; figures regenerate from data with one script.

PHASE 5 — INKLING SCALE
- NVFP4 inference paths on B200; multi-node (2x8 B200) with GPU-direct RDMA KV transfer
  (NCCL/NVSHMEM) and prefill/decode disaggregation
- Run Inkling-Small 276B first, then Inkling 975B; map Inkling's thinking_effort parameter
  (0.2..0.99) to tree width/depth; add all rows to ThoughtBench
GATE: Inkling rows live on the leaderboard with reproducible configs.

PHASE 6 — ENTERPRISE PACKAGING
- Helm chart + K8s operator (autoscale on KV pressure); SLURM templates; on-prem/VPC docs
- OIDC/SAML SSO, RBAC, audit logs, per-tenant quotas; Grafana dashboard pack
- vLLM/SGLang flag-compatibility mode; HF + S3 model registry support
GATE: clean install on a fresh cluster in under 30 minutes following the docs.

DEFINITION OF DONE
All phase gates pass; README quickstart under 5 minutes; one command reproduces the headline
AIME result; repo is Apache-2.0, CI green, results JSON public.
10. This week
Feed the E2E prompt to your builder agent — Phase 1 starts today
Reserve ThoughtBench domain + GitHub org now
Draft the Inkling bench plan in parallel — first to publish its deep-reasoning cost curve owns the conversation
Wire AutoKernel in: every tree trace trains it to generate better tree-attention kernels — engine + compiler + flywheel = uncopiable moat