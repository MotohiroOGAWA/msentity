from __future__ import annotations

import argparse
from pathlib import Path

from msentity import MSDataset, load_ms_dataset


SUPPORTED_SUFFIXES = {
    ".msp",
    ".mgf",
    ".msds",
    ".hdf5",
    ".h5",
}


def setup_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "merge-dir",
        help="Merge MS dataset files in a directory into a .msds dataset.",
    )

    parser.add_argument(
        "input_dir",
        help="Input directory containing dataset files.",
    )

    parser.add_argument(
        "output_file",
        help="Output .msds dataset file path.",
    )

    parser.add_argument(
        "--file-type",
        default=None,
        choices=["msp", "mgf", "msds"],
        help="Input file type. If omitted, files are inferred from their extensions.",
    )

    parser.add_argument(
        "--pattern",
        default=None,
        help="Glob pattern used to select input files. Defaults to supported dataset files.",
    )

    parser.add_argument(
        "--recursive",
        nargs="?",
        type=_positive_int,
        default=None,
        metavar="DEPTH",
        help=(
            "Search input files recursively. If DEPTH is omitted, all "
            "subdirectories are searched. DEPTH=1 searches only input_dir; "
            "DEPTH=2 includes direct subdirectories."
        ),
    )

    parser.add_argument(
        "--add-source",
        action="store_true",
        help="Add source columns: path and source_index within the input file.",
    )

    parser.add_argument(
        "--spec-id-prefix",
        default=None,
        help="Prefix used to generate SpecID when needed.",
    )

    parser.add_argument(
        "--description",
        default="",
        help="Description for the merged dataset.",
    )

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if output_file.suffix.lower() != ".msds":
        raise ValueError("Currently, merge-dir supports output to .msds only.")

    input_files = _collect_input_files(
        input_dir,
        file_type=args.file_type,
        pattern=args.pattern,
        max_depth=args.recursive,
    )

    if not input_files:
        raise ValueError(f"No input dataset files found in directory: {input_dir}")

    datasets = []
    for input_file in input_files:
        dataset = load_ms_dataset(
            input_file,
            file_type=args.file_type,
            spec_id_prefix=args.spec_id_prefix,
        )

        if args.add_source:
            dataset["path"] = input_file.relative_to(input_dir).as_posix()
            dataset["source_index"] = list(range(len(dataset)))

        datasets.append(dataset)

    merged = MSDataset.concat(datasets, description=args.description)
    merged.save(str(output_file))

    print(f"Merged {len(input_files)} files into: {output_file}")


def _collect_input_files(
    input_dir: Path,
    *,
    file_type: str | None,
    pattern: str | None,
    max_depth: int | None,
) -> list[Path]:
    if pattern is not None:
        candidates = _glob(input_dir, pattern)
    elif file_type is not None:
        candidates = _glob(input_dir, f"*.{file_type}")
    else:
        candidates = [
            path
            for path in _glob(input_dir, "*")
            if path.suffix.lower() in SUPPORTED_SUFFIXES
        ]

    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and _within_depth(path, input_dir, max_depth=max_depth)
        ),
        key=lambda path: path.relative_to(input_dir).as_posix(),
    )


def _glob(
    input_dir: Path,
    pattern: str,
) -> list[Path]:
    return list(input_dir.rglob(pattern))


def _within_depth(
    path: Path,
    input_dir: Path,
    *,
    max_depth: int | None,
) -> bool:
    if max_depth is None:
        return True

    return len(path.relative_to(input_dir).parts) <= max_depth


def _positive_int(value: str) -> int:
    number = int(value)

    if number < 1:
        raise argparse.ArgumentTypeError("DEPTH must be 1 or greater.")

    return number
