"""Command-line entry point for autotree-serve."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import uvicorn

from .app import create_app
from .engine import DeterministicEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autotree")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser(
        "serve",
        help="Run the OpenAI-compatible API server.",
        description=(
            "Run autotree-serve. The deterministic engine is a seeded toy generator "
            "for API development and does not serve real model weights."
        ),
    )
    serve.add_argument("--model", required=True, help="Model identifier exposed by the API.")
    serve.add_argument(
        "--engine",
        default="deterministic",
        help=(
            "Engine implementation. 'deterministic' is a seeded toy generator; "
            "'treekv' is reserved for Phase 2 integration."
        ),
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.engine == "treekv":
        print(
            "Tree-KV engine integration lands in Phase 2; the scheduler, GPU runtime, "
            "and real-model weight loader are not present in autotree-serve yet.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.engine != "deterministic":
        print(f"Unknown engine '{args.engine}'. No fallback was selected.", file=sys.stderr)
        raise SystemExit(2)

    engine = DeterministicEngine(model_id=args.model)
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
