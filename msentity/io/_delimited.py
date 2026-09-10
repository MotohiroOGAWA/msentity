from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..core.MSDataset import MSDataset
from ..core.PeakSeries import PeakSeries
from ..processing.id import set_spec_id


PEAK_COLUMN = "Peak"


def _parse_peaks(value: str, row_number: int, format_name: str) -> list[tuple[float, float]]:
    peaks: list[tuple[float, float]] = []
    if not value.strip():
        return peaks
    for peak_number, item in enumerate(value.split(";"), start=1):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid Peak value at {format_name} row {row_number}, peak {peak_number}: "
                "expected 'mz,intensity'"
            )
        try:
            peaks.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric Peak value at {format_name} row {row_number}, peak {peak_number}: {item}"
            ) from exc
    return peaks


def _infer_metadata(rows: list[dict[str, str]], columns: list[str]) -> pd.DataFrame:
    metadata = pd.DataFrame(rows, columns=columns)
    for column in columns:
        values = metadata[column].replace("", None)
        nonempty = values.notna()
        numeric = pd.to_numeric(values, errors="coerce")
        metadata[column] = numeric if numeric[nonempty].notna().all() else values
    return metadata


def read_delimited_text(
    text: str,
    *,
    delimiter: str,
    format_name: str,
    source_name: str = "<delimited_text>",
    spec_id_prefix: str | None = None,
) -> MSDataset:
    """Read delimited text containing metadata columns and a ``Peak`` column."""
    if not text or not text.strip():
        raise ValueError(f"{format_name} text is empty.")

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        raise ValueError(f"{format_name} header is missing.")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError(f"{format_name} column names must be unique.")
    if PEAK_COLUMN not in reader.fieldnames:
        raise ValueError(f"{format_name} must contain a '{PEAK_COLUMN}' column.")

    metadata_columns = [column for column in reader.fieldnames if column != PEAK_COLUMN]
    metadata_rows: list[dict[str, str]] = []
    peak_rows: list[tuple[float, float]] = []
    offsets = [0]
    for row_number, row in enumerate(reader, start=2):
        if None in row:
            raise ValueError(f"Too many fields at {format_name} row {row_number} in {source_name}.")
        metadata_rows.append({column: row.get(column, "") or "" for column in metadata_columns})
        peaks = _parse_peaks(row.get(PEAK_COLUMN, "") or "", row_number, format_name)
        peak_rows.extend(peaks)
        offsets.append(len(peak_rows))

    peak_data = np.asarray(peak_rows, dtype=float).reshape((-1, 2))
    dataset = MSDataset(
        _infer_metadata(metadata_rows, metadata_columns),
        PeakSeries(peak_data, np.asarray(offsets, dtype=np.int64)),
    )
    if spec_id_prefix is not None and "SpecID" not in dataset.columns:
        set_spec_id(dataset, prefix=spec_id_prefix)
    return dataset


def read_delimited(
    filepath: str | Path,
    *,
    delimiter: str,
    format_name: str,
    encoding: str = "utf-8-sig",
    spec_id_prefix: str | None = None,
    show_progress: bool = True,
) -> MSDataset:
    """Read a delimited spectrum table from a file."""
    del show_progress  # Kept consistent with the other text readers.
    path = Path(filepath)
    return read_delimited_text(
        path.read_text(encoding=encoding),
        delimiter=delimiter,
        format_name=format_name,
        source_name=str(path),
        spec_id_prefix=spec_id_prefix,
    )


def write_delimited(
    dataset: MSDataset,
    path: str | Path,
    *,
    delimiter: str,
    format_name: str,
    headers: Sequence[str] | None = None,
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> None:
    """Write one spectrum per delimited row using ``mz,intensity;...`` in ``Peak``."""
    del show_progress  # Kept consistent with the other writers.
    selected = list(dataset.columns if headers is None else headers)
    if PEAK_COLUMN in selected:
        raise ValueError(f"'{PEAK_COLUMN}' is reserved for spectrum peaks in {format_name} files.")
    missing = [column for column in selected if column not in dataset.columns]
    if missing:
        raise ValueError(f"Unknown {format_name} metadata columns: {missing}")

    with Path(path).open("w", encoding=encoding, newline="") as stream:
        writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
        writer.writerow([*selected, PEAK_COLUMN])
        for record in dataset:
            peak_text = ";".join(
                f"{format(peak.mz, '.17g')},{format(peak.intensity, '.17g')}"
                for peak in record.peaks
            )
            writer.writerow([*(record[column] for column in selected), peak_text])
