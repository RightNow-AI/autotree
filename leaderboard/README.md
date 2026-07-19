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
`thoughtbench.results.v2` contract with Zod, and exports the site and downloadable
`out/data.json`. Missing provenance, missing required fields, schema drift, and
unknown fields stop the build. The fixture warning shown on every page is derived
from the validated provenance in those result files. The superseded v1 contract
is rejected rather than interpreted with v2 metric semantics.

## Refreshing fixtures

The current v2 contract deliberately accepts only `kind: "fixture"`, requires
the fixture notice, and requires `benchmark_claims_allowed: false`. Real results
must not be introduced by relabeling or hand-editing these files.

1. Run both bundled ThoughtBench demo configurations against the deterministic
   fixture server and copy their emitted `*.results.json` artifacts into
   `fixtures/`; do not edit versions, metric values, or provenance by hand.
2. Confirm every artifact still carries the fixture provenance notice and
   `benchmark_claims_allowed: false`.
3. Run `pnpm test` and `pnpm build`, then inspect the static export before making
   any claim about the fixture data.
