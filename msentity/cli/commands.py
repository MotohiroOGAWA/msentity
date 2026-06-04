from __future__ import annotations

import argparse
import json
from pathlib import Path

from .shell import run_shell

from msentity import MSDataset, load_ms_dataset

def add_shell_command(
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
        choices=["msp", "mgf", "msds"],
        help="Input file type. If omitted, it is inferred from the file extension.",
    )

    parser.add_argument(
        "--spec-id-prefix",
        default=None,
        help="Prefix used to generate SpecID when needed.",
    )

    parser.set_defaults(func=run_shell_command)


def run_shell_command(args: argparse.Namespace) -> None:
    run_shell(
        args.input_file,
        file_type=args.file_type,
        spec_id_prefix=args.spec_id_prefix,
    )

def add_info_command(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "info",
        help="Show summary information about an MS dataset.",
    )

    add_input_dataset_arguments(parser)

    parser.set_defaults(func=run_info)


def add_head_command(
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

    parser.set_defaults(func=run_head)


def add_convert_command(
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

    parser.set_defaults(func=run_convert)


def add_meta_command(
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

    parser.set_defaults(func=run_meta)


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


def run_info(args: argparse.Namespace) -> None:
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


def run_head(args: argparse.Namespace) -> None:
    dataset = load_ms_dataset(
        args.input_file,
        file_type=args.file_type,
        spec_id_prefix=args.spec_id_prefix,
    )

    metadata = dataset.metadata.head(args.num_rows)
    print(metadata.to_string(index=True))


def run_convert(args: argparse.Namespace) -> None:
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


def run_meta(args: argparse.Namespace) -> None:
    meta = MSDataset.read_dataset_meta(args.input_file)

    result = {
        "description": meta.description,
        "attributes": meta.attributes,
        "tags": meta.tags,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))