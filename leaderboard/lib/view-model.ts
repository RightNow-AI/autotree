import type { EngineMode, ResultsDocument } from "./results-schema";
import { modelSlug } from "./results";

export type ChartPoint = {
  id: string;
  model: string;
  modelSlug: string;
  taskSet: string;
  mode: EngineMode;
  budget: string;
  accuracyK: number;
  accuracy: number;
  meanCostUsd: number;
};

function mean(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function highestAccuracyEntry(
  accuracyAtK: ResultsDocument["aggregate_metrics"][number]["accuracy_at_k"],
) {
  const entries = Object.entries(accuracyAtK).sort(
    ([left], [right]) => Number(left) - Number(right),
  );
  const [key, value] = entries.at(-1) ?? ["1", { mean: null }];
  return { k: Number(key), accuracy: value.mean ?? 0 };
}

export function chartPoints(results: ResultsDocument[]): ChartPoint[] {
  return results.flatMap((result) =>
    result.aggregate_metrics.map((aggregate) => {
      const seedCells = result.per_seed_metrics.filter(
        (cell) => cell.budget_name === aggregate.budget_name,
      );
      if (seedCells.length === 0) {
        throw new Error(
          `No seed metrics for ${result.engine_config.model}/${aggregate.budget_name}`,
        );
      }
      const accuracy = highestAccuracyEntry(aggregate.accuracy_at_k);
      return {
        id: `${result.run_id}:${aggregate.budget_name}`,
        model: result.engine_config.model,
        modelSlug: modelSlug(result.engine_config.model),
        taskSet: result.task_set.name,
        mode: result.engine_config.mode,
        budget: aggregate.budget_name,
        accuracyK: accuracy.k,
        accuracy: accuracy.accuracy,
        meanCostUsd: mean(
          seedCells.map((cell) => cell.metrics.total_cost_usd),
        ),
      };
    }),
  );
}

export function distinctModels(results: ResultsDocument[]) {
  return [...new Set(results.map((result) => result.engine_config.model))].sort();
}

export function distinctTaskSets(results: ResultsDocument[]) {
  return [...new Set(results.map((result) => result.task_set.name))].sort();
}
