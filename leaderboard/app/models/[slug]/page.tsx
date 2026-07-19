import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { loadResults, modelSlug } from "../../../lib/results";
import { formatKvReuseRatio } from "../../../lib/view-model";

export const dynamicParams = false;

function formatNumber(value: number | null, digits = 2) {
  return value === null ? "not reported" : value.toFixed(digits);
}

function formatPercent(value: number | null) {
  return value === null ? "not reported" : `${(value * 100).toFixed(1)}%`;
}

function formatUsd(value: number | null) {
  if (value === null) return "not available (no correct fixture samples)";
  return value < 0.01 ? `$${value.toFixed(6)}` : `$${value.toFixed(2)}`;
}

function modeLabel(mode: "sequential" | "tree") {
  return mode === "tree" ? "AutoTree ON / tree" : "AutoTree OFF / sequential";
}

export function generateStaticParams() {
  return [
    ...new Set(loadResults().map((result) => modelSlug(result.engine_config.model))),
  ].map((slug) => ({ slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const result = loadResults().find(
    (candidate) => modelSlug(candidate.engine_config.model) === slug,
  );
  return { title: result ? `${result.engine_config.model} · ThoughtBench fixture` : "Model not found" };
}

export default async function ModelPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const documents = loadResults().filter(
    (result) => modelSlug(result.engine_config.model) === slug,
  );
  if (documents.length === 0) notFound();

  const model = documents[0].engine_config.model;
  const seedRows = documents.flatMap((document) =>
    document.per_seed_metrics.map((cell) => ({ document, cell })),
  );

  return (
    <article className="model-detail">
      <Link className="back-link" href="/">← Back to leaderboard</Link>
      <div className="detail-heading">
        <div>
          <p className="eyebrow">Fixture model identifier</p>
          <h1>{model}</h1>
          <p>
            Values below come directly from deterministic fixture result files.
            Nulls remain explicit; no unavailable metric is estimated.
          </p>
        </div>
        <a className="button" href="/data.json" download>Download all JSON</a>
      </div>

      <section aria-labelledby="run-configs">
        <h2 id="run-configs">Run configurations</h2>
        <div className="run-grid">
          {documents.map((document) => (
            <article className="run-card" key={document.run_id}>
              <span className={`mode-badge ${document.engine_config.mode}`}>
                {modeLabel(document.engine_config.mode)}
              </span>
              <dl>
                <div><dt>Task set</dt><dd>{document.task_set.name}</dd></div>
                <div><dt>Seeds</dt><dd>{document.engine_config.seeds.join(", ")}</dd></div>
                <div><dt>Samples</dt><dd>{document.samples.length}</dd></div>
                <div><dt>k samples</dt><dd>{document.engine_config.k_samples}</dd></div>
                <div><dt>Input price / 1M</dt><dd>{formatUsd(document.engine_config.pricing.input_per_million_usd)}</dd></div>
                <div><dt>Output price / 1M</dt><dd>{formatUsd(document.engine_config.pricing.output_per_million_usd)}</dd></div>
              </dl>
            </article>
          ))}
        </div>
      </section>

      <section aria-labelledby="seed-metrics">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Metrics and token accounting</p>
            <h2 id="seed-metrics">Per-seed breakdown</h2>
          </div>
        </div>
        <div className="table-shell">
          <table>
            <thead>
              <tr>
                <th>Mode</th><th>Budget</th><th>Seed</th><th>Accuracy@1</th>
                <th>Input tokens</th><th>Output tokens</th><th>Total cost</th>
                <th>Cost / correct</th><th>Tokens / sec</th><th>TTFT</th>
                <th>KV reuse</th><th>Useful tokens</th>
              </tr>
            </thead>
            <tbody>
              {seedRows.map(({ document, cell }) => (
                <tr key={`${document.run_id}:${cell.protocol_seed}:${cell.budget_name}`}>
                  <td>{modeLabel(document.engine_config.mode)}</td>
                  <td>{cell.budget_name}</td>
                  <td>{cell.protocol_seed}</td>
                  <td>{formatPercent(cell.metrics.accuracy_at_k["1"] ?? null)}</td>
                  <td>{cell.metrics.input_tokens}</td>
                  <td>{cell.metrics.output_tokens}</td>
                  <td>{formatUsd(cell.metrics.total_cost_usd)}</td>
                  <td>{formatUsd(cell.metrics.cost_per_correct_usd)}</td>
                  <td>{formatNumber(cell.metrics.tokens_per_second.mean)}</td>
                  <td>{cell.metrics.ttft_seconds.mean === null ? "not reported" : `${formatNumber(cell.metrics.ttft_seconds.mean, 4)} s`}</td>
                  <td>{formatKvReuseRatio(cell.metrics.kv_reuse_ratio.mean)}</td>
                  <td>{formatPercent(cell.metrics.useful_token_ratio.mean)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </article>
  );
}
