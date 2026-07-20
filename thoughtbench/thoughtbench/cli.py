"""ThoughtBench command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .report import render_report
from .runner import load_run_config, run_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thoughtbench")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run or resume a fixture benchmark")
    run.add_argument("--config", type=Path, required=True)
    report = commands.add_parser("report", help="render a results JSON table")
    report.add_argument("results", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        config = load_run_config(args.config)
        results = run_benchmark(config)
        print(f"wrote {config.output_path} ({len(results.samples)} samples)")
        return 0
    print(render_report(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
