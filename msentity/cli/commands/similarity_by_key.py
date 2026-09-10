import argparse
from pathlib import Path

from msentity import load_ms_dataset
from msentity.cli.commands._similarity import (
    add_similarity_calculation_arguments,
    add_similarity_output_arguments,
)
from msentity.similarity import calculate_similarity


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "similarity-by-key",
        help="Compare spectra for unique keys shared by two datasets.",
    )
    parser.add_argument("input1")
    parser.add_argument("input2")
    parser.add_argument("--key1", default="SpecID")
    parser.add_argument("--key2", default="SpecID")
    add_similarity_calculation_arguments(parser)
    add_similarity_output_arguments(parser, embed_by_default=False)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    result = calculate_similarity(
        load_ms_dataset(args.input1), load_ms_dataset(args.input2),
        source1=str(Path(args.input1).resolve()), source2=str(Path(args.input2).resolve()),
        key1=args.key1, key2=args.key2, method=args.method, bin_width=args.bin_width,
        intensity_exponent=args.intensity_exponent, max_cum_peaks=args.max_cum_peaks,
        include_matched_data=args.include_matched_data,
    )
    result.save(args.output)
    print(f"Saved {len(result.table)} similarities: {args.output}")
