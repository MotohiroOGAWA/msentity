from __future__ import annotations

import argparse

from msentity.cli.commands import add_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msentity",
        description="Command line interface for msentity.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    add_commands(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
