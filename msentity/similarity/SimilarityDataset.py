"""SimilarityDataset: HDF5 persistence and tabular result operations."""
from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ..core.MSDataset import MSDataset
from .calculation import cosine_similarity_by_key


@dataclass
class SimilarityDataset:
    """A table of spectrum similarities and its calculation metadata."""

    table: pd.DataFrame
    metadata: dict[str, Any]

    _REQUIRED_COLUMNS = {"index1", "index2", "cosine_similarity"}

    def __post_init__(self) -> None:
        if not isinstance(self.table, pd.DataFrame):
            raise TypeError("table must be a pandas.DataFrame")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        missing = self._REQUIRED_COLUMNS.difference(self.table.columns)
        if missing:
            raise ValueError(f"Similarity table is missing required columns: {sorted(missing)}")
        scores = self.table["cosine_similarity"]
        if not pd.api.types.is_numeric_dtype(scores):
            raise ValueError("Similarity scores must be numeric")
        numeric_scores = scores.to_numpy(dtype=float)
        if not np.isfinite(numeric_scores).all() or not scores.between(0, 1).all():
            raise ValueError("Similarity scores must be finite values between 0 and 1")

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return f"SimilarityDataset(n_pairs={len(self)}, columns={self.columns})"

    @property
    def columns(self) -> list[str]:
        return self.table.columns.tolist()

    @property
    def n_pairs(self) -> int:
        return len(self)

    def copy(self) -> SimilarityDataset:
        """Return an independent copy of the result table and metadata."""
        import copy

        return SimilarityDataset(self.table.copy(deep=True), copy.deepcopy(self.metadata))

    def filter(self, mask: Any) -> SimilarityDataset:
        """Return a result containing rows selected by a boolean mask."""
        filtered = self.table.loc[mask].reset_index(drop=True)
        metadata = dict(self.metadata)
        metadata["row_count"] = len(filtered)
        metadata["source_row_count"] = len(self.table)
        return SimilarityDataset(filtered, metadata)

    def sort_values(
        self, by: str | list[str], *, ascending: bool | list[bool] = True
    ) -> SimilarityDataset:
        """Return a stably sorted similarity result."""
        return SimilarityDataset(
            self.table.sort_values(by=by, ascending=ascending, kind="mergesort")
            .reset_index(drop=True),
            dict(self.metadata),
        )

    def describe_scores(self) -> dict[str, float]:
        """Return count and five-number/central-tendency score statistics."""
        scores = self.table["cosine_similarity"]
        return {
            "count": float(scores.count()),
            "mean": float(scores.mean()),
            "q1": float(scores.quantile(0.25)),
            "median": float(scores.median()),
            "q3": float(scores.quantile(0.75)),
            "min": float(scores.min()),
            "max": float(scores.max()),
        }

    def histogram(self, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Return score frequencies and bin edges over the fixed range 0–1."""
        if not isinstance(bins, (int, np.integer)) or not 1 <= bins <= 200:
            raise ValueError("bins must be an integer between 1 and 200")
        return np.histogram(
            self.table["cosine_similarity"].to_numpy(), bins=int(bins), range=(0, 1)
        )

    def export_table(self, path: str | Path) -> None:
        """Export the result table as Parquet, CSV, or TSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".parquet":
            self.table.to_parquet(path, index=False)
        elif path.suffix.lower() == ".csv":
            self.table.to_csv(path, index=False)
        elif path.suffix.lower() == ".tsv":
            self.table.to_csv(path, index=False, sep="\t")
        else:
            raise ValueError("Table export format must be parquet, csv, or tsv")

    def save(self, path: str | Path) -> None:
        """Atomically write a .mssim file; preserve an existing file on failure."""
        path = Path(path)
        if path.suffix.lower() != ".mssim":
            raise ValueError("Similarity files must use the .mssim extension")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.table.to_parquet(index=False)
        metadata = json.dumps(self.metadata, ensure_ascii=False, allow_nan=False)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(descriptor)
        try:
            with h5py.File(temporary, "w") as handle:
                handle.attrs["format"] = "msentity.similarity"
                handle.attrs["schema_version"] = 1
                handle.create_dataset("table.parquet", data=np.frombuffer(payload, dtype=np.uint8))
                handle.create_dataset("metadata.json", data=metadata,
                                      dtype=h5py.string_dtype("utf-8"))
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path) -> SimilarityDataset:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("format") != "msentity.similarity":
                raise ValueError("Not an msentity similarity file")
            if handle.attrs.get("schema_version") != 1:
                raise ValueError("Unsupported similarity schema version")
            table = pd.read_parquet(io.BytesIO(handle["table.parquet"][()].tobytes()))
            metadata = json.loads(handle["metadata.json"].asstr()[()])
        return cls(table, metadata)

    @classmethod
    def from_datasets(
        cls,
        ds1: MSDataset,
        ds2: MSDataset,
        **kwargs: Any,
    ) -> SimilarityDataset:
        """Calculate a similarity dataset from two mass-spectrum datasets."""
        return calculate_similarity(ds1, ds2, **kwargs)


def calculate_similarity(ds1: MSDataset, ds2: MSDataset, *, source1: str = "", source2: str = "",
                         key1: str = "SpecID", key2: str = "SpecID",
                         bin_width: float = 0.01, intensity_exponent: float = 1.0,
                         max_cum_peaks: int = 200_000) -> SimilarityDataset:
    parameters = dict(key1=key1, key2=key2, bin_width=bin_width,
                      intensity_exponent=intensity_exponent, max_cum_peaks=max_cum_peaks)
    table = cosine_similarity_by_key(ds1, ds2, **parameters)
    sources = [dict(path=path, n_rows=ds.n_rows, description=ds.description,
                    attributes=ds.attributes, tags=ds.tags)
               for ds, path in ((ds1, source1), (ds2, source2))]
    return SimilarityDataset(table, {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "binned_cosine_similarity_by_key", "parameters": parameters,
        "matching": "unique_intersection", "sources": sources,
        "row_count": len(table), "index_basis": "zero-based input dataset position",
        "input_scope": "entire in-memory datasets (including unsaved edits)",
    })


# Compatibility alias for callers that used the initial result class name.
SimilarityResult = SimilarityDataset
