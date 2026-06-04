from __future__ import annotations

import argparse

from msentity.cli.commands import (
    add_convert_command,
    add_head_command,
    add_info_command,
    add_meta_command,
    add_shell_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msentity",
        description="Command line interface for msentity.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    add_info_command(subparsers)
    add_head_command(subparsers)
    add_convert_command(subparsers)
    add_meta_command(subparsers)
    add_shell_command(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()