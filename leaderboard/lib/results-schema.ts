export type ResultsDocument = Record<string, unknown>;

export function validateResultsDocument(input: unknown): ResultsDocument {
  // RED-first pass-through: the strict ThoughtBench schema is implemented next.
  return input as ResultsDocument;
}
