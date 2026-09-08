import argparse
from pathlib import Path

from msentity import load_ms_dataset
from msentity.similarity import SimilarityDataset


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "similarity-by-key",
        help="Compare spectra for unique keys shared by two datasets.",
    )
    parser.add_argument("input1")
    parser.add_argument("input2")
    parser.add_argument("--key1", default="SpecID")
    parser.add_argument("--key2", default="SpecID")
    parser.add_argument("--bin-width", type=float, default=0.01)
    parser.add_argument("--intensity-exponent", type=float, default=1.0)
    parser.add_argument("--max-cum-peaks", type=int, default=200_000)
    parser.add_argument("--output", required=True, help="Output .mssim file")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    result = SimilarityDataset.from_datasets(
        load_ms_dataset(args.input1), load_ms_dataset(args.input2),
        source1=str(Path(args.input1).resolve()), source2=str(Path(args.input2).resolve()),
        key1=args.key1, key2=args.key2, bin_width=args.bin_width,
        intensity_exponent=args.intensity_exponent, max_cum_peaks=args.max_cum_peaks,
    )
    result.save(args.output)
    print(f"Saved {len(result.table)} similarities: {args.output}")
