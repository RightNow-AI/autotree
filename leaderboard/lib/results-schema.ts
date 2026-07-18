import { z } from "zod";

export const RESULTS_SCHEMA_VERSION = "thoughtbench.results.v1";
export const FIXTURE_NOTICE =
  "FIXTURE TASKS ONLY - NOT A REAL BENCHMARK RESULT.";

const strictObject = <T extends z.ZodRawShape>(shape: T) =>
  z.object(shape).strict();

const fixtureProvenanceSchema = strictObject({
  kind: z.literal("fixture"),
  source: z.string().min(1),
  license: z.string().min(1),
  notice: z.literal(FIXTURE_NOTICE),
});

const pricingConfigSchema = strictObject({
  input_per_million_usd: z.number().nonnegative(),
  output_per_million_usd: z.number().nonnegative(),
});

const treeConfigSchema = strictObject({
  policy: z.enum(["beam", "best_first", "mcts"]),
  branches: z.number().int().min(1),
  scorer: z.string().nullable(),
});

const budgetConfigSchema = strictObject({
  name: z.string().min(1).regex(/^[A-Za-z0-9_.-]+$/),
  max_tokens: z.number().int().min(1),
  tree_budget_tokens: z.number().int().min(1).nullable(),
});

const numericStatsSchema = strictObject({
  count: z.number().int().nonnegative(),
  mean: z.number().nullable(),
  minimum: z.number().nullable(),
  maximum: z.number().nullable(),
  spread: z.number().nullable(),
  p50: z.number().nullable(),
  p95: z.number().nullable(),
});

const aggregateValueSchema = strictObject({
  count: z.number().int().nonnegative(),
  mean: z.number().nullable(),
  minimum: z.number().nullable(),
  maximum: z.number().nullable(),
  spread: z.number().nullable(),
});

const metricSetSchema = strictObject({
  task_count: z.number().int().min(1),
  sample_count: z.number().int().min(1),
  correct_sample_count: z.number().int().nonnegative(),
  accuracy_at_k: z.record(z.string(), z.number()),
  pass_power_k: z.record(z.string(), z.number()),
  input_tokens: z.number().int().nonnegative(),
  output_tokens: z.number().int().nonnegative(),
  tokens_per_correct: z.number().nullable(),
  total_cost_usd: z.number().nonnegative(),
  cost_per_correct_usd: z.number().nullable(),
  latency_seconds: numericStatsSchema,
  ttft_seconds: numericStatsSchema,
  tokens_per_second: numericStatsSchema,
  rollout_throughput_per_hour: numericStatsSchema,
  kv_reuse_ratio: numericStatsSchema,
  useful_token_ratio: numericStatsSchema,
});

const seedMetricsSchema = strictObject({
  protocol_seed: z.number().int(),
  budget_name: z.string(),
  metrics: metricSetSchema,
});

const aggregateMetricsSchema = strictObject({
  budget_name: z.string(),
  seed_count: z.number().int().min(1),
  accuracy_at_k: z.record(z.string(), aggregateValueSchema),
  pass_power_k: z.record(z.string(), aggregateValueSchema),
  tokens_per_correct: aggregateValueSchema,
  cost_per_correct_usd: aggregateValueSchema,
  latency_mean_seconds: aggregateValueSchema,
  ttft_mean_seconds: aggregateValueSchema,
  tokens_per_second_mean: aggregateValueSchema,
  rollout_throughput_per_hour_mean: aggregateValueSchema,
  kv_reuse_ratio_mean: aggregateValueSchema,
  useful_token_ratio_mean: aggregateValueSchema,
});

const treeStatsSchema = strictObject({
  policy: z.string().nullable(),
  branch_count: z.number().int().nonnegative(),
  pruned_count: z.number().int().nonnegative(),
  merged_count: z.number().int().nonnegative().nullable(),
  winner_branch_id: z.string().nullable(),
  tokens_spent_per_branch: z.record(z.string(), z.number().int()),
  final_scores: z.union([
    z.record(z.string(), z.number()),
    z.array(z.number()),
  ]),
  scorer: z.string().nullable(),
});

const sampleResultSchema = strictObject({
  sample_key: z.string(),
  task_id: z.string(),
  protocol_seed: z.number().int(),
  request_seed: z.number().int(),
  budget_name: z.string(),
  sample_index: z.number().int().nonnegative(),
  response_text: z.string(),
  expected_answer: z.string(),
  grader: z.enum(["exact-match", "numeric"]),
  tags: z.array(z.string()),
  correct: z.boolean(),
  prompt_tokens: z.number().int().nonnegative(),
  completion_tokens: z.number().int().nonnegative(),
  total_tokens: z.number().int().nonnegative(),
  latency_seconds: z.number().nonnegative(),
  ttft_seconds: z.number().nonnegative().nullable(),
  tokens_per_second: z.number().nonnegative().nullable(),
  rollout_throughput_per_hour: z.number().nonnegative().nullable(),
  kv_reuse_ratio: z.number().min(0).max(1).nullable(),
  useful_token_ratio: z.number().min(0).max(1).nullable(),
  tree: treeStatsSchema.nullable(),
});

const engineConfigSchema = strictObject({
  model: z.string(),
  base_url: z.string().url(),
  mode: z.enum(["sequential", "tree"]),
  budgets: z.array(budgetConfigSchema),
  k_samples: z.number().int().min(1).max(64),
  seeds: z.array(z.number().int()),
  concurrency: z.number().int().min(1),
  temperature: z.number().min(0).max(2),
  top_p: z.number().gt(0).max(1),
  stop: z.array(z.string()),
  tree: treeConfigSchema.nullable(),
  pricing: pricingConfigSchema,
});

const taskSetSchema = strictObject({
  name: z.string(),
  sha256: z.string().regex(/^[0-9a-f]{64}$/),
  task_count: z.number().int().min(1),
  provenance: fixtureProvenanceSchema,
});

const environmentSchema = strictObject({
  generated_at_utc: z.string(),
  python: z.string(),
  platform: z.string(),
  thoughtbench_version: z.string(),
  autotree_sdk_version: z.string(),
  autotree_serve_version: z.string().nullable(),
});

export const resultsDocumentSchema = strictObject({
  schema_version: z.literal(RESULTS_SCHEMA_VERSION),
  artifact_notice: z.literal(FIXTURE_NOTICE),
  benchmark_claims_allowed: z.literal(false),
  run_id: z.string(),
  run_fingerprint: z.string().regex(/^[0-9a-f]{64}$/),
  engine_config: engineConfigSchema,
  task_set: taskSetSchema,
  per_seed_metrics: z.array(seedMetricsSchema),
  aggregate_metrics: z.array(aggregateMetricsSchema),
  samples: z.array(sampleResultSchema),
  environment: environmentSchema,
});

export type ResultsDocument = z.infer<typeof resultsDocumentSchema>;
export type EngineMode = ResultsDocument["engine_config"]["mode"];

export class ResultsValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ResultsValidationError";
  }
}

export function validateResultsDocument(input: unknown): ResultsDocument {
  const result = resultsDocumentSchema.safeParse(input);
  if (result.success) {
    return result.data;
  }

  const details = result.error.issues
    .map((issue) => `${issue.path.join(".") || "document"}: ${issue.message}`)
    .join("; ");
  throw new ResultsValidationError(`Invalid ThoughtBench results: ${details}`);
}
