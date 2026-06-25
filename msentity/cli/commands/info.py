from __future__ import annotations

import argparse
import json

from msentity import load_ms_dataset
from msentity.cli.commands._common import add_input_dataset_arguments


def setup_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "info",
        help="Show summary information about an MS dataset.",
    )

    add_input_dataset_arguments(parser)

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    dataset = load_ms_dataset(
        args.input_file,
        file_type=args.file_type,
        spec_id_prefix=args.spec_id_prefix,
    )

    summary = {
        "input_file": args.input_file,
        "n_spectra": len(dataset),
        "n_columns": dataset.n_columns,
        "n_peaks_total": dataset.n_peaks_total,
        "columns": dataset.columns,
        "description": dataset.description,
        "attributes": dataset.attributes,
        "tags": dataset.tags,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
