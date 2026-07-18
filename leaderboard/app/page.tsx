import Link from "next/link";

import { LeaderboardExplorer } from "../components/leaderboard-explorer";
import { loadResults, modelSlug } from "../lib/results";
import {
  chartPoints,
  distinctModels,
  distinctTaskSets,
} from "../lib/view-model";

export default function HomePage() {
  const results = loadResults();
  const models = distinctModels(results);

  return (
    <>
      <section className="hero">
        <p className="eyebrow">ThoughtBench · contract preview</p>
        <h1>Benchmark data, rendered without claims.</h1>
        <p className="hero-copy">
          This static export exercises the complete results pipeline using only
          the deterministic fixtures emitted by ThoughtBench. It is not evidence
          of model quality, cost savings, or production performance.
        </p>
        <div className="hero-actions">
          <a className="button primary" href="/data.json" download>Download validated data</a>
          <span>{results.length} fixture result files · {models.length} fixture model identifier</span>
        </div>
      </section>

      <LeaderboardExplorer
        points={chartPoints(results)}
        taskSets={distinctTaskSets(results)}
      />

      <section className="model-index" aria-labelledby="models-heading">
        <p className="eyebrow">Model pages</p>
        <h2 id="models-heading">Inspect fixture runs by model identifier</h2>
        <div className="model-grid">
          {models.map((model) => {
            const modelResults = results.filter(
              (result) => result.engine_config.model === model,
            );
            return (
              <Link className="model-card" href={`/models/${modelSlug(model)}`} key={model}>
                <span>{model}</span>
                <small>{modelResults.map((result) => result.engine_config.mode).sort().join(" + ")}</small>
              </Link>
            );
          })}
        </div>
      </section>
    </>
  );
}
