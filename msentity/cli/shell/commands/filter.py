from __future__ import annotations

import operator
from typing import Callable

import pandas as pd

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class FilterCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="filter",
            usage=("filter <column> <op> <value> | filter peak <min> <max> "
                   "[intensity] | filter neutral-loss <min> <max> [intensity] "
                   "[precursor_column]"),
            summary="Filter spectra by metadata or spectral peaks.",
            description=(
                "Filter spectra by a metadata column. "
                "The filtered result becomes the current dataset view."
            ),
            arguments=[
                (
                    "column",
                    "Metadata column name.",
                ),
                (
                    "op",
                    "Comparison operator. Supported operators are: ==, !=, >, >=, <, <=, contains.",
                ),
                (
                    "value",
                    "Value to compare against.",
                ),
            ],
            examples=[
                "filter PrecursorMZ > 300",
                "filter Name contains glucose",
                "filter IonMode == POSITIVE",
                "filter AdductType == [M+H]+",
                "filter peak 100.0 100.1 1000",
                "filter neutral-loss 17.9 18.1 1000",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if args and args[0] == "peak":
            if len(args) not in {3, 4}:
                raise ValueError("Usage: filter peak <min> <max> [intensity]")
            state.dataset = state.dataset.filter_by_peak(
                float(args[1]), float(args[2]), float(args[3]) if len(args) == 4 else 0.0
            )
            print(f"Filtered dataset: {len(state.dataset)} spectra")
            return True

        if args and args[0] in {"neutral-loss", "neutral_loss"}:
            if len(args) not in {3, 4, 5}:
                raise ValueError(
                    "Usage: filter neutral-loss <min> <max> [intensity] [precursor_column]"
                )
            state.dataset = state.dataset.filter_by_neutral_loss(
                float(args[1]),
                float(args[2]),
                float(args[3]) if len(args) >= 4 else 0.0,
                precursor_mz_column=args[4] if len(args) == 5 else "PrecursorMZ",
            )
            print(f"Filtered dataset: {len(state.dataset)} spectra")
            return True

        if len(args) < 3:
            raise ValueError("Usage: filter <column> <op> <value>")

        column = args[0]
        op = args[1]
        value = " ".join(args[2:])

        dataset = state.dataset

        if column not in dataset.columns:
            raise KeyError(f"column not found: {column}")

        series = dataset[column]
        mask = build_filter_mask(
            series=series,
            op=op,
            value=value,
        )

        state.dataset = dataset[mask.reset_index(drop=True)]

        print(f"Filtered dataset: {len(state.dataset)} spectra")

        return True


def build_filter_mask(
    *,
    series: pd.Series,
    op: str,
    value: str,
) -> pd.Series:
    if op == "contains":
        return series.astype(str).str.contains(
            value,
            case=False,
            na=False,
            regex=False,
        )

    operators: dict[str, Callable[[pd.Series, object], pd.Series]] = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
    }

    if op not in operators:
        raise ValueError(
            "Unsupported operator. Use one of: ==, !=, >, >=, <, <=, contains"
        )

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )
    numeric_value = try_parse_float(value)

    if numeric_value is not None and numeric_series.notna().any():
        return operators[op](
            numeric_series,
            numeric_value,
        ).fillna(False)

    return operators[op](
        series.astype(str),
        value,
    ).fillna(False)


def try_parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def get_command() -> ShellCommand:
    return FilterCommand()
