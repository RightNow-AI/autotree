import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { validateResultsDocument } from "../lib/results-schema";

const fixturePath = resolve(
  process.cwd(),
  "fixtures",
  "demo-sequential.results.json",
);
const treeFixturePath = resolve(
  process.cwd(),
  "fixtures",
  "demo-tree.results.json",
);

function fixtureCopy(): Record<string, any> {
  return JSON.parse(readFileSync(fixturePath, "utf8"));
}

function treeFixtureCopy(): Record<string, any> {
  return JSON.parse(readFileSync(treeFixturePath, "utf8"));
}

function expectKeys(value: Record<string, unknown>, keys: string[]) {
  expect(Object.keys(value).sort()).toEqual([...keys].sort());
}

describe("ThoughtBench results contract", () => {
  it("accepts the fixture emitted by ThoughtBench", () => {
    expect(validateResultsDocument(fixtureCopy())).toBeTruthy();
  });

  it("pins every top-level field in thoughtbench.results.v1", () => {
    expectKeys(fixtureCopy(), [
      "aggregate_metrics",
      "artifact_notice",
      "benchmark_claims_allowed",
      "engine_config",
      "environment",
      "per_seed_metrics",
      "run_fingerprint",
      "run_id",
      "samples",
      "schema_version",
      "task_set",
    ]);
  });

  it("pins nested fields emitted by ThoughtBench", () => {
    const document = fixtureCopy();
    const treeDocument = treeFixtureCopy();

    expectKeys(document.engine_config, [
      "base_url",
      "budgets",
      "concurrency",
      "k_samples",
      "mode",
      "model",
      "pricing",
      "seeds",
      "stop",
      "temperature",
      "top_p",
      "tree",
    ]);
    expectKeys(document.engine_config.budgets[0], [
      "max_tokens",
      "name",
      "tree_budget_tokens",
    ]);
    expectKeys(document.engine_config.pricing, [
      "input_per_million_usd",
      "output_per_million_usd",
    ]);
    expectKeys(document.task_set, [
      "name",
      "provenance",
      "sha256",
      "task_count",
    ]);
    expectKeys(document.task_set.provenance, [
      "kind",
      "license",
      "notice",
      "source",
    ]);
    expectKeys(document.per_seed_metrics[0], [
      "budget_name",
      "metrics",
      "protocol_seed",
    ]);
    expectKeys(document.per_seed_metrics[0].metrics, [
      "accuracy_at_k",
      "correct_sample_count",
      "cost_per_correct_usd",
      "input_tokens",
      "kv_reuse_ratio",
      "latency_seconds",
      "output_tokens",
      "pass_power_k",
      "rollout_throughput_per_hour",
      "sample_count",
      "task_count",
      "tokens_per_correct",
      "tokens_per_second",
      "total_cost_usd",
      "ttft_seconds",
      "useful_token_ratio",
    ]);
    expectKeys(document.per_seed_metrics[0].metrics.latency_seconds, [
      "count",
      "maximum",
      "mean",
      "minimum",
      "p50",
      "p95",
      "spread",
    ]);
    expectKeys(document.aggregate_metrics[0], [
      "accuracy_at_k",
      "budget_name",
      "cost_per_correct_usd",
      "kv_reuse_ratio_mean",
      "latency_mean_seconds",
      "pass_power_k",
      "rollout_throughput_per_hour_mean",
      "seed_count",
      "tokens_per_correct",
      "tokens_per_second_mean",
      "ttft_mean_seconds",
      "useful_token_ratio_mean",
    ]);
    expectKeys(document.aggregate_metrics[0].tokens_per_correct, [
      "count",
      "maximum",
      "mean",
      "minimum",
      "spread",
    ]);
    expectKeys(document.samples[0], [
      "budget_name",
      "completion_tokens",
      "correct",
      "expected_answer",
      "grader",
      "kv_reuse_ratio",
      "latency_seconds",
      "prompt_tokens",
      "protocol_seed",
      "request_seed",
      "response_text",
      "rollout_throughput_per_hour",
      "sample_index",
      "sample_key",
      "tags",
      "task_id",
      "tokens_per_second",
      "total_tokens",
      "tree",
      "ttft_seconds",
      "useful_token_ratio",
    ]);
    expectKeys(treeDocument.samples[0].tree, [
      "branch_count",
      "final_scores",
      "merged_count",
      "policy",
      "pruned_count",
      "scorer",
      "tokens_spent_per_branch",
      "winner_branch_id",
    ]);
    expectKeys(document.environment, [
      "autotree_sdk_version",
      "autotree_serve_version",
      "generated_at_utc",
      "platform",
      "python",
      "thoughtbench_version",
    ]);
  });

  it("rejects a results file without provenance", () => {
    const document = fixtureCopy();
    delete document.task_set.provenance;

    expect(() => validateResultsDocument(document)).toThrow(
      /task_set\.provenance/,
    );
  });

  it("rejects schema drift when a required field is renamed", () => {
    const document = fixtureCopy();
    document.schemaVersion = document.schema_version;
    delete document.schema_version;

    expect(() => validateResultsDocument(document)).toThrow(/schema_version/);
  });
});
