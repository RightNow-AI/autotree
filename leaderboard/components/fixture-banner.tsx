import type { ResultsDocument } from "../lib/results-schema";

type FixtureBannerProps = {
  provenances: ResultsDocument["task_set"]["provenance"][];
};

export function FixtureBanner({ provenances }: FixtureBannerProps) {
  const labels = [
    ...new Set(
      provenances.map((provenance) =>
        provenance.kind === "fixture"
          ? "FIXTURE DATA - not real benchmark results"
          : provenance.kind,
      ),
    ),
  ];
  const notices = [
    ...new Set(provenances.map((provenance) => provenance.notice)),
  ];

  return (
    <aside className="fixture-banner" role="alert">
      <strong>{labels.join(" / ")}</strong>
      <span>{notices.join(" · ")}</span>
    </aside>
  );
}
