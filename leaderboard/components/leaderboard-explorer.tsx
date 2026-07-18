"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import type { ChartPoint } from "../lib/view-model";

type ModeFilter = "all" | "sequential" | "tree";

type LeaderboardExplorerProps = {
  points: ChartPoint[];
  taskSets: string[];
};

const modeCopy = {
  all: "Compare both",
  sequential: "AutoTree OFF",
  tree: "AutoTree ON",
} as const;

function usd(value: number) {
  if (value === 0) return "$0";
  if (value < 0.01) return `$${value.toFixed(6)}`;
  return `$${value.toFixed(2)}`;
}

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function LeaderboardExplorer({
  points,
  taskSets,
}: LeaderboardExplorerProps) {
  const [mode, setMode] = useState<ModeFilter>("all");
  const [taskSet, setTaskSet] = useState("all");

  const visiblePoints = useMemo(
    () =>
      points.filter(
        (point) =>
          (mode === "all" || point.mode === mode) &&
          (taskSet === "all" || point.taskSet === taskSet),
      ),
    [mode, points, taskSet],
  );

  const width = 900;
  const height = 440;
  const margin = { top: 30, right: 40, bottom: 70, left: 76 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maximumCost = Math.max(...visiblePoints.map((point) => point.meanCostUsd), 1e-6);
  const xMaximum = maximumCost * 1.12;
  const x = (value: number) => margin.left + (value / xMaximum) * plotWidth;
  const y = (value: number) =>
    margin.top + plotHeight - Math.max(0, Math.min(1, value)) * plotHeight;
  const xTicks = [0, 0.25, 0.5, 0.75, 1];
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <section className="explorer" aria-labelledby="chart-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Accuracy vs. cost</p>
          <h2 id="chart-heading">Fixture result explorer</h2>
        </div>
        <div className="filters">
          <label>
            Task set
            <select value={taskSet} onChange={(event) => setTaskSet(event.target.value)}>
              <option value="all">All task sets</option>
              {taskSets.map((name) => (
                <option value={name} key={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <div className="mode-filter" aria-label="Execution mode filter">
            {(Object.keys(modeCopy) as ModeFilter[]).map((value) => (
              <button
                className={mode === value ? "active" : ""}
                key={value}
                onClick={() => setMode(value)}
                type="button"
              >
                {modeCopy[value]}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="chart-shell">
        <svg
          className="scatter"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-labelledby="scatter-title scatter-description"
        >
          <title id="scatter-title">Accuracy versus benchmark run cost</title>
          <desc id="scatter-description">
            Fixture points grouped by sequential AutoTree off and tree AutoTree on modes.
          </desc>
          {yTicks.map((tick) => (
            <g key={`y-${tick}`}>
              <line
                className="grid-line"
                x1={margin.left}
                x2={margin.left + plotWidth}
                y1={y(tick)}
                y2={y(tick)}
              />
              <text className="axis-tick" x={margin.left - 14} y={y(tick) + 4} textAnchor="end">
                {percent(tick)}
              </text>
            </g>
          ))}
          {xTicks.map((tick) => (
            <g key={`x-${tick}`}>
              <line
                className="grid-line"
                x1={x(xMaximum * tick)}
                x2={x(xMaximum * tick)}
                y1={margin.top}
                y2={margin.top + plotHeight}
              />
              <text
                className="axis-tick"
                x={x(xMaximum * tick)}
                y={margin.top + plotHeight + 26}
                textAnchor="middle"
              >
                {usd(xMaximum * tick)}
              </text>
            </g>
          ))}
          <text
            className="axis-label"
            x={margin.left + plotWidth / 2}
            y={height - 12}
            textAnchor="middle"
          >
            Mean fixture run cost per seed (USD)
          </text>
          <text
            className="axis-label"
            transform={`translate(20 ${margin.top + plotHeight / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            Accuracy@k
          </text>
          {visiblePoints.map((point) => (
            <g key={point.id}>
              <circle
                className={`data-point ${point.mode}`}
                cx={x(point.meanCostUsd)}
                cy={y(point.accuracy)}
                r="10"
              >
                <title>
                  {`${point.model} · ${modeCopy[point.mode]} · ${point.budget} · accuracy@${point.accuracyK} ${percent(point.accuracy)} · ${usd(point.meanCostUsd)}`}
                </title>
              </circle>
            </g>
          ))}
        </svg>
        <div className="legend" aria-label="Chart legend">
          <span><i className="legend-dot sequential" />AutoTree OFF / sequential</span>
          <span><i className="legend-dot tree" />AutoTree ON / tree</span>
        </div>
      </div>

      <div className="point-list" aria-label="Visible chart values">
        {visiblePoints.map((point) => (
          <article className="point-card" key={point.id}>
            <div>
              <span className={`mode-badge ${point.mode}`}>{modeCopy[point.mode]}</span>
              <h3><Link href={`/models/${point.modelSlug}`}>{point.model}</Link></h3>
              <p>{point.taskSet} · {point.budget}</p>
            </div>
            <dl>
              <div><dt>Accuracy@{point.accuracyK}</dt><dd>{percent(point.accuracy)}</dd></div>
              <div><dt>Mean cost / seed</dt><dd>{usd(point.meanCostUsd)}</dd></div>
            </dl>
          </article>
        ))}
      </div>
    </section>
  );
}
