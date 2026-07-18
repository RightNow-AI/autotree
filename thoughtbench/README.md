# ThoughtBench

ThoughtBench is a fixture-first benchmark harness. The bundled tasks are tiny,
synthetic contract fixtures; they are not AIME, GPQA, LiveCodeBench, or evidence
for any performance, accuracy, or cost claim. Every v1 results artifact carries
the stamp `FIXTURE TASKS ONLY - NOT A REAL BENCHMARK RESULT.` and disallows
benchmark claims.

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
validated against the versioned `thoughtbench.results.v1` JSON Schema before an
atomic replace.

`accuracy@k` means the fraction of tasks with at least one correct result in the
first k samples. `pass_power_k` is the stricter fraction whose first k samples
all pass. Requests for either metric raise if any task has fewer than k samples.

Tree mode uses the SDK's typed `/v1/tree/completions` call. The current SDK/server
non-stream response exposes usage and branch summaries but not TTFT, KV reuse, or
useful-token counters, so those fields remain explicit `null`/zero-count metrics
unless an endpoint reports them.
