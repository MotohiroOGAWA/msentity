from __future__ import annotations

import argparse
from pathlib import Path

from msentity import load_ms_dataset
from msentity.cli.commands._common import add_input_dataset_arguments


def setup_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "convert",
        help="Convert an MS dataset to .msds format.",
    )

    add_input_dataset_arguments(parser)

    parser.add_argument(
        "output_file",
        help="Output dataset file path.",
    )

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    dataset = load_ms_dataset(
        args.input_file,
        file_type=args.file_type,
        spec_id_prefix=args.spec_id_prefix,
    )

    output_file = Path(args.output_file)

    if output_file.suffix.lower() != ".msds":
        raise ValueError("Currently, convert supports output to .msds only.")

    dataset.save(str(output_file))

    print(f"Saved: {output_file}")
