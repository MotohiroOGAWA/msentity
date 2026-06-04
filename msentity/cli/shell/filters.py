from __future__ import annotations

import operator
from typing import Callable

import pandas as pd


def build_filter_mask(
    *,
    series: pd.Series,
    op: str,
    value: str,
) -> pd.Series:
    """Build a boolean mask for filtering a metadata column."""
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
    """Try to parse a string as float."""
    try:
        return float(value)
    except ValueError:
        return None