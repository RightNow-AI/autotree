import { describe, expect, it } from "vitest";

import { formatKvReuseRatio } from "../lib/view-model";

describe("leaderboard metric formatting", () => {
  it("renders KV reuse as a multiplier rather than a percentage", () => {
    expect(formatKvReuseRatio(3.5)).toBe("3.5x");
    expect(formatKvReuseRatio(5)).toBe("5.0x");
    expect(formatKvReuseRatio(null)).toBe("not reported");
  });
});
