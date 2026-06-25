from __future__ import annotations

import argparse


def add_input_dataset_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "input_file",
        help="Input dataset file path.",
    )

    parser.add_argument(
        "--file-type",
        default=None,
        choices=["msp", "mgf", "msds"],
        help="Input file type. If omitted, it is inferred from the file extension.",
    )

    parser.add_argument(
        "--spec-id-prefix",
        default=None,
        help="Prefix used to generate SpecID when needed.",
    )
