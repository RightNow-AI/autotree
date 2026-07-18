# ThoughtBench leaderboard

Static Next.js renderer for versioned ThoughtBench results artifacts. The
committed data is deterministic fixture data, not benchmark evidence.

From this directory:

```console
pnpm install --frozen-lockfile
pnpm test
pnpm build
```

The build reads every `fixtures/*.results.json` file, validates the complete
`thoughtbench.results.v1` contract with Zod, and exports the site and downloadable
`out/data.json`. Missing provenance, missing required fields, schema drift, and
unknown fields stop the build. The fixture warning shown on every page is derived
from the validated provenance in those result files.

## Replacing fixtures with real results

The current contract deliberately accepts only `kind: "fixture"`, requires the
fixture notice, and requires `benchmark_claims_allowed: false`. Real results must
not be introduced by relabeling or hand-editing these files.

When ThoughtBench defines a versioned real-result provenance contract:

1. Run the ThoughtBench harness and copy its emitted `*.results.json` artifacts
   into `fixtures/`; do not edit metric values by hand.
2. Mirror that new versioned contract in `lib/results-schema.ts`, including its
   provenance and claims rules.
3. Update the field-level compatibility and rejection tests in `tests/` and the
   provenance-driven banner behavior in the same change.
4. Run `pnpm test` and `pnpm build`, then inspect the static export before making
   any benchmark claim.
