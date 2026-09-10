"""Spectrum tables in CSV format."""
from pathlib import Path
from typing import Sequence

from ..core.MSDataset import MSDataset
from ._delimited import read_delimited, read_delimited_text, write_delimited


def read_csv_text(
    text: str, *, source_name: str = "<csv_text>", spec_id_prefix: str | None = None,
) -> MSDataset:
    """Read CSV text with metadata columns and a Peak column."""
    return read_delimited_text(
        text, delimiter=",", format_name="CSV",
        source_name=source_name, spec_id_prefix=spec_id_prefix,
    )


def read_csv(
    filepath: str | Path, *, encoding: str = "utf-8-sig",
    spec_id_prefix: str | None = None, show_progress: bool = True,
) -> MSDataset:
    """Read a CSV spectrum table."""
    return read_delimited(
        filepath, delimiter=",", format_name="CSV", encoding=encoding,
        spec_id_prefix=spec_id_prefix, show_progress=show_progress,
    )


def write_csv(
    dataset: MSDataset, path: str | Path, *, headers: Sequence[str] | None = None,
    encoding: str = "utf-8", show_progress: bool = True,
) -> None:
    """Write one spectrum per CSV row with mz,intensity pairs in Peak."""
    write_delimited(
        dataset, path, delimiter=",", format_name="CSV",
        headers=headers, encoding=encoding, show_progress=show_progress,
    )
