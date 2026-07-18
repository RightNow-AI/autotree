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
            "Run autotree-serve. 'deterministic' does not serve real model weights; "
            "it is a seeded toy generator. "
            "'treekv' loads a real HuggingFace model through the CPU Tree-KV demo engine."
        ),
    )
    serve.add_argument(
        "--model",
        default="gpt2",
        help="Model identifier exposed by the API (default: gpt2).",
    )
    serve.add_argument(
        "--engine",
        default="deterministic",
        help=(
            "Engine implementation. 'deterministic' is a seeded toy generator; "
            "'treekv' uses the Rust scheduler and real HuggingFace weights on CPU."
        ),
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.engine == "treekv":
        try:
            engine = _load_treekv_engine(args.model)
        except Exception as error:
            print(
                f"Failed to load Tree-KV CPU model {args.model!r}: {error}",
                file=sys.stderr,
            )
            raise SystemExit(2) from error
    elif args.engine == "deterministic":
        engine = DeterministicEngine(model_id=args.model)
    else:
        print(f"Unknown engine '{args.engine}'. No fallback was selected.", file=sys.stderr)
        raise SystemExit(2)

    uvicorn.run(create_app(engine), host=args.host, port=args.port)


def _load_treekv_engine(model_id: str):
    from autotree_core.engine import TreeKVEngine

    return TreeKVEngine(model_id=model_id)


if __name__ == "__main__":
    main()
