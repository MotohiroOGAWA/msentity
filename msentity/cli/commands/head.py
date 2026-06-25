from __future__ import annotations

import argparse

from msentity import load_ms_dataset
from msentity.cli.commands._common import add_input_dataset_arguments


def setup_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "head",
        help="Show the first rows of spectrum metadata.",
    )

    add_input_dataset_arguments(parser)

    parser.add_argument(
        "-n",
        "--num-rows",
        type=int,
        default=5,
        help="Number of rows to show.",
    )

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    dataset = load_ms_dataset(
        args.input_file,
        file_type=args.file_type,
        spec_id_prefix=args.spec_id_prefix,
    )

    metadata = dataset.metadata.head(args.num_rows)
    print(metadata.to_string(index=True))
