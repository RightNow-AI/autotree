# ThoughtBench

ThoughtBench runs honest, reproducible reasoning benchmarks against any
OpenAI-compatible HTTP endpoint. One configuration selects a `single`,
`best_of_n`, or `tree` arm. Each output contains per-task measurements, a
regime stamp, Wilson confidence bounds, and only metrics that were actually
observed.

Existing measured artifacts in `results/` use the earlier provenance schema
and are preserved unchanged. The aggregator ignores those legacy shapes and
merges current harness artifacts only.

## Install and run

ThoughtBench requires Python 3.12. From the repository root, install it into a
Python environment and run:

```console
python -m pip install -e thoughtbench
python -m thoughtbench run --config thoughtbench/configs/autotree_tree.json
python -m thoughtbench aggregate thoughtbench/results
```

JSON and YAML configurations are supported. Set `OPENAI_API_KEY` when an
endpoint requires bearer authentication, or set `api_key_env` in the config to
the name of another environment variable.

The bundled endpoint values use the reserved `.invalid` domain. Replace the
model and endpoint before a real run.

## Dry run

Dry run reads and validates the config and task file, prints every exact URL
and JSON request body, performs no HTTP request, and writes no result file:

```console
python -m thoughtbench run \
  --config thoughtbench/configs/vllm_bestofn.json \
  --dry-run
```

## Arms

- `single` sends one `/v1/chat/completions` request with one choice.
- `best_of_n` sends one `/v1/chat/completions` request with `n` choices and
  performs normalized majority voting in the client.
- `tree` sends one `/v1/tree/completions` request with `policy`, `branches`,
  and `budget_tokens` in the `tree` object.

Every task is run once for every configured seed. Concurrency applies across
those task and seed executions. Request failures stop the run rather than
creating a partial public artifact.

## Grading and measurements

Numeric answer extraction follows the engine conventions: the last boxed
value wins, then the value after `####`, then the last final-answer marker,
then the last numeric token. Normalization removes commas, surrounding
whitespace, and one terminal period. Best-of-n voting uses the same normalized
answers.

Completion tokens are read from the endpoint. Tree token cost is the sum of
`tree.tokens_spent_per_branch`, with `usage.completion_tokens` used only when
the tree breakdown is absent. A task's token field is `null` when neither
measurement exists. Aggregate token statistics are `null` if any task lacks a
token measurement. Wall time is measured by the client for every request.

Each current result has this direct chart input shape:

```text
meta: engine label, model, redacted endpoint, arm, parameters, seeds, git SHA, start time
tasks: id, seed, correct, extracted answer, gold answer, tokens, wall seconds
summary: accuracy, 95% Wilson bounds, token statistics, tokens per correct, task count
```

Results are written to `results/<timestamp>-<label>.json`. Aggregation writes
`leaderboard.json` by default and retains each source filename, metadata stamp,
and summary.

## Task sets

`tasks/math12.jsonl` contains the twelve in-house arithmetic checks used by the
earlier ad hoc benchmark.

`tasks/gsm8k_subset.jsonl` contains the first 50 records of the official GSM8K
test split, with the gold value taken from each record's `####` answer marker.
The source dataset is the OpenAI `grade-school-math` repository.

Standard citation:

> Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun,
> Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
> Nakano, Christopher Hesse, and John Schulman. "Training Verifiers to Solve
> Math Word Problems." arXiv:2110.14168, 2021.

## Tests

The full package suite is CPU-only and does not call an external service:

```console
core/.venv/Scripts/python.exe -m pytest thoughtbench -q
```

HTTP behavior is covered with mocked transports. The retained integration test
starts a deterministic loopback server in-process.
