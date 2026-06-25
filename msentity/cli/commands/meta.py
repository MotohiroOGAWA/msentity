from __future__ import annotations

import argparse
import json

from msentity import MSDataset


def setup_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "meta",
        help="Show metadata of an .msds dataset.",
    )

    parser.add_argument(
        "input_file",
        help="Input .msds file path.",
    )

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    meta = MSDataset.read_dataset_meta(args.input_file)

    result = {
        "description": meta.description,
        "attributes": meta.attributes,
        "tags": meta.tags,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
