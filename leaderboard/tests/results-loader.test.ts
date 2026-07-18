import {
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { loadResultsFromDirectory } from "../lib/results";

const temporaryDirectories: string[] = [];

function temporaryResultsDirectory() {
  const directory = mkdtempSync(join(tmpdir(), "thoughtbench-results-"));
  temporaryDirectories.push(directory);
  return directory;
}

function fixtureCopy(): Record<string, any> {
  return JSON.parse(
    readFileSync(
      resolve(process.cwd(), "fixtures", "demo-sequential.results.json"),
      "utf8",
    ),
  );
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("results directory loader", () => {
  it("loads and validates every vendored results document", () => {
    const results = loadResultsFromDirectory(
      resolve(process.cwd(), "fixtures"),
    );

    expect(results).toHaveLength(2);
    expect(results.map((result) => result.engine_config.mode).sort()).toEqual([
      "sequential",
      "tree",
    ]);
  });

  it("fails at load time when a results file lacks provenance", () => {
    const directory = temporaryResultsDirectory();
    const document = fixtureCopy();
    delete document.task_set.provenance;
    writeFileSync(
      join(directory, "invalid.results.json"),
      JSON.stringify(document),
      "utf8",
    );

    expect(() => loadResultsFromDirectory(directory)).toThrow(
      /invalid\.results\.json.*task_set\.provenance/,
    );
  });
});
