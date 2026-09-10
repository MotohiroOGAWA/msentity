from __future__ import annotations

import argparse
from pathlib import Path

from msentity import load_ms_dataset
from msentity.cli.commands._similarity import (
    add_similarity_calculation_arguments,
    add_similarity_output_arguments,
)
from msentity.similarity import calculate_library_search


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "library-search",
        help="Compare every query spectrum with every reference spectrum.",
    )
    parser.add_argument("query", help="Query dataset file")
    parser.add_argument("reference", help="Reference library dataset file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum score retained in the result (default: 0.8).",
    )
    parser.add_argument(
        "--max-pairs-per-call",
        type=int,
        default=2_000_000,
        help="Maximum candidate pairs generated in one outer chunk.",
    )
    add_similarity_calculation_arguments(parser)
    add_similarity_output_arguments(parser, embed_by_default=True)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    query_path = Path(args.query).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()
    result = calculate_library_search(
        load_ms_dataset(query_path),
        load_ms_dataset(reference_path),
        query_source=str(query_path),
        reference_source=str(reference_path),
        threshold=args.threshold,
        method=args.method,
        bin_width=args.bin_width,
        intensity_exponent=args.intensity_exponent,
        max_pairs_per_call=args.max_pairs_per_call,
        max_cum_peaks=args.max_cum_peaks,
        include_matched_data=args.include_matched_data,
        show_progress=True,
    )
    result.save(args.output)
    print(f"Saved {len(result.table)} library matches: {args.output}")
