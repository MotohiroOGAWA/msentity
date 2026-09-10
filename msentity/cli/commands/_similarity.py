from __future__ import annotations

import argparse


def add_similarity_calculation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add calculation options shared by key matching and library search."""
    parser.add_argument(
        "--method",
        choices=["cosine", "reverse_cosine"],
        default="cosine",
        help="Similarity calculation method (default: cosine).",
    )
    parser.add_argument("--bin-width", type=float, default=0.01)
    parser.add_argument("--intensity-exponent", type=float, default=1.0)
    parser.add_argument("--max-cum-peaks", type=int, default=200_000)


def add_similarity_output_arguments(
    parser: argparse.ArgumentParser,
    *,
    embed_by_default: bool,
) -> None:
    """Add the shared MSSIM output and matched-data storage choices."""
    parser.add_argument("--output", required=True, help="Output .mssim file")
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument(
        "--include-matched-data",
        dest="include_matched_data",
        action="store_true",
        help="Embed every uniquely matched spectrum and metadata record once.",
    )
    storage.add_argument(
        "--lightweight",
        dest="include_matched_data",
        action="store_false",
        help="Save only match indices, scores, and calculation metadata.",
    )
    parser.set_defaults(include_matched_data=embed_by_default)
