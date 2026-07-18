import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { validateResultsDocument } from "../lib/results-schema";

const fixturePath = resolve(
  process.cwd(),
  "fixtures",
  "demo-sequential.results.json",
);

function fixtureCopy(): Record<string, any> {
  return JSON.parse(readFileSync(fixturePath, "utf8"));
}

describe("ThoughtBench results contract", () => {
  it("accepts the fixture emitted by ThoughtBench", () => {
    expect(validateResultsDocument(fixtureCopy())).toBeTruthy();
  });

  it("pins every top-level field in thoughtbench.results.v1", () => {
    expect(Object.keys(fixtureCopy()).sort()).toEqual([
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
