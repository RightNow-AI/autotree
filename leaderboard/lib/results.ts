import { existsSync, readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

import {
  type ResultsDocument,
  ResultsValidationError,
  validateResultsDocument,
} from "./results-schema";

export const DEFAULT_RESULTS_DIRECTORY = resolve(process.cwd(), "fixtures");

export function loadResultsFromDirectory(
  directory = DEFAULT_RESULTS_DIRECTORY,
): ResultsDocument[] {
  const files = readdirSync(directory)
    .filter((name) => name.endsWith(".results.json"))
    .sort();

  if (files.length === 0) {
    throw new ResultsValidationError(
      `No ThoughtBench *.results.json files found in ${directory}`,
    );
  }

  return files.map((file) => {
    const path = resolve(directory, file);
    let payload: unknown;
    try {
      payload = JSON.parse(readFileSync(path, "utf8"));
    } catch (error) {
      throw new ResultsValidationError(
        `Unable to parse ${file}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }

    try {
      return validateResultsDocument(payload);
    } catch (error) {
      throw new ResultsValidationError(
        `${file}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  });
}

export const REAL_RESULTS_DIRECTORY = resolve(process.cwd(), "results");

export function loadResults(): ResultsDocument[] {
  const documents = loadResultsFromDirectory();
  if (existsSync(REAL_RESULTS_DIRECTORY)) {
    documents.push(...loadResultsFromDirectory(REAL_RESULTS_DIRECTORY));
  }
  return documents;
}

export function modelSlug(model: string): string {
  return model
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}
