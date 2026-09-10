from __future__ import annotations

import argparse

from msentity.cli.shell.runner import DatasetShell, run_shell


def setup_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "shell",
        help="Start a dataset shell.",
    )

    parser.add_argument(
        "input_file",
        help="Input dataset file path.",
    )

    parser.add_argument(
        "--file-type",
        default=None,
        choices=["msp", "mgf", "msds", "tsv", "csv"],
        help="Input file type. If omitted, it is inferred from the file extension.",
    )

    parser.add_argument(
        "--spec-id-prefix",
        default=None,
        help="Prefix used to generate SpecID when needed.",
    )

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    run_shell(
        args.input_file,
        file_type=args.file_type,
        spec_id_prefix=args.spec_id_prefix,
    )


__all__ = [
    "DatasetShell",
    "run_shell",
    "setup_parser",
]
