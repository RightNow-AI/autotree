import { loadResults } from "../../lib/results";

export const dynamic = "force-static";

export function GET() {
  return new Response(JSON.stringify(loadResults(), null, 2), {
    headers: {
      "content-disposition": 'attachment; filename="thoughtbench-results.json"',
      "content-type": "application/json; charset=utf-8",
    },
  });
}
