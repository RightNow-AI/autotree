import Link from "next/link";

export default function NotFound() {
  return (
    <section className="empty-state">
      <p className="eyebrow">Not found</p>
      <h1>No fixture model matches this route.</h1>
      <Link className="button" href="/">Return to leaderboard</Link>
    </section>
  );
}
