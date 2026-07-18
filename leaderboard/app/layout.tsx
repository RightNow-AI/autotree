import type { Metadata } from "next";
import Link from "next/link";

import { FixtureBanner } from "../components/fixture-banner";
import { loadResults } from "../lib/results";

import "./globals.css";

export const metadata: Metadata = {
  title: "ThoughtBench fixture leaderboard",
  description: "Static fixture-data renderer for ThoughtBench result artifacts.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const results = loadResults();
  const provenances = results.map((result) => result.task_set.provenance);

  return (
    <html lang="en">
      <body>
        <FixtureBanner provenances={provenances} />
        <header className="site-header">
          <Link className="brand" href="/">ThoughtBench</Link>
          <nav aria-label="Primary navigation">
            <Link href="/">Leaderboard</Link>
            <a href="/data.json" download>Download JSON</a>
          </nav>
        </header>
        <main>{children}</main>
        <footer>
          <p>Fixture-only static renderer for the versioned ThoughtBench results contract.</p>
        </footer>
      </body>
    </html>
  );
}
