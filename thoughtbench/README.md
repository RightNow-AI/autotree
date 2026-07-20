# ThoughtBench

ThoughtBench is a benchmark harness with schema-enforced task provenance.
Fixture task sets produce artifacts stamped
`FIXTURE TASKS ONLY - NOT A REAL BENCHMARK RESULT.` with benchmark claims
disallowed. Real task sets (declared with source and license) produce
artifacts stamped as measured results with claims allowed, limited to the
stated protocol scope. The first real run ships in `results/`; see
`docs/first-benchmark.md` at the repository root.

The package uses editable local path dependencies so the harness exercises the
repository's real SDK and test server:

```toml
[tool.uv.sources]
autotree-sdk = { path = "../sdk", editable = true }
autotree-serve = { path = "../serve", editable = true }
```

From this directory:

```console
uv run pytest -q
uv run uvicorn autotree_serve.app:create_app --factory --host 127.0.0.1 --port 8000
```

With that deterministic fixture server running, use another terminal:

```console
uv run thoughtbench run --config fixtures/demo-sequential.json
uv run thoughtbench report fixtures/demo-sequential.results.json
```

Run configuration is JSON. It fixes exactly three protocol seeds, one or more
named token budgets, k samples per task, decoding settings, concurrency, pricing,
and sequential or tree execution. A sibling append-only `.partial.jsonl` journal
is fsynced after every sample and reused after interruption. The final JSON is
validated against the versioned `thoughtbench.results.v2` JSON Schema before an
atomic replace.

`accuracy@k` means the fraction of tasks with at least one correct result in the
first k samples. `pass_power_k` is the stricter fraction whose first k samples
all pass. Requests for either metric raise if any task has fewer than k samples.

Tree mode uses the SDK's typed `/v1/tree/completions` call. `kv_reuse_ratio` is
the core logical-token count divided by the physical-token count and is therefore
`>= 1`; ThoughtBench preserves that value without percentage-style clamping.
TTFT and useful-token ratios remain explicit `null`/zero-count metrics unless an
endpoint reports them.
