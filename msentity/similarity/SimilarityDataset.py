"""SimilarityDataset: HDF5 persistence and tabular result operations."""
from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pandas as pd

from ..core.MSDataset import MSDataset, SpectrumRecord
from .calculation import library_search, similarity_by_key


@dataclass
class SimilarityDataset:
    """A similarity table with calculation metadata and optional matched data."""

    table: pd.DataFrame
    metadata: dict[str, Any]
    matched_datasets: tuple[MSDataset, MSDataset] | None = None
    matched_source_indices: tuple[np.ndarray, np.ndarray] | None = None

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
        matched_datasets = self.matched_datasets
        matched_source_indices = self.matched_source_indices
        if (matched_datasets is None) != (matched_source_indices is None):
            raise ValueError("matched datasets and source indices must be provided together")
        if matched_datasets is not None and matched_source_indices is not None:
            for side in (0, 1):
                data_column = f"data_index{side + 1}"
                if data_column not in self.table:
                    raise ValueError(f"Embedded matches require the '{data_column}' column")
                source_indices = np.asarray(matched_source_indices[side], dtype=np.int64)
                if source_indices.ndim != 1 or len(source_indices) != len(matched_datasets[side]):
                    raise ValueError("Embedded source indices must match their datasets")
                data_indices = self.table[data_column].to_numpy(dtype=np.int64)
                if data_indices.size and (
                    data_indices.min() < 0 or data_indices.max() >= len(source_indices)
                ):
                    raise ValueError(f"'{data_column}' contains an invalid embedded index")
            self.matched_source_indices = (
                np.asarray(matched_source_indices[0], dtype=np.int64),
                np.asarray(matched_source_indices[1], dtype=np.int64),
            )

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return f"SimilarityDataset(n_pairs={len(self)}, columns={self.columns})"

    @property
    def columns(self) -> list[str]:
        return [str(column) for column in self.table.columns]

    @property
    def n_pairs(self) -> int:
        return len(self)

    @property
    def has_matched_data(self) -> bool:
        """Whether unique matched spectra and metadata are embedded."""
        return self.matched_datasets is not None

    def copy(self) -> SimilarityDataset:
        """Return an independent copy of the result table and metadata."""
        import copy

        datasets = None
        source_indices = None
        if self.matched_datasets is not None and self.matched_source_indices is not None:
            datasets = (
                self.matched_datasets[0].copy(),
                self.matched_datasets[1].copy(),
            )
            source_indices = (
                self.matched_source_indices[0].copy(),
                self.matched_source_indices[1].copy(),
            )
        return SimilarityDataset(
            self.table.copy(deep=True),
            copy.deepcopy(self.metadata),
            datasets,
            source_indices,
        )

    def filter(self, mask: Any) -> SimilarityDataset:
        """Return a result containing rows selected by a boolean mask."""
        filtered = self.table.loc[mask].reset_index(drop=True)
        metadata = dict(self.metadata)
        metadata["row_count"] = len(filtered)
        metadata["source_row_count"] = len(self.table)
        return SimilarityDataset(
            filtered,
            metadata,
            self.matched_datasets,
            self.matched_source_indices,
        )

    def sort_values(
        self, by: str | list[str], *, ascending: bool | list[bool] = True
    ) -> SimilarityDataset:
        """Return a stably sorted similarity result."""
        return SimilarityDataset(
            self.table.sort_values(by=by, ascending=ascending, kind="mergesort")
            .reset_index(drop=True),
            dict(self.metadata),
            self.matched_datasets,
            self.matched_source_indices,
        )

    def match_records(self, row: int) -> tuple[SpectrumRecord, SpectrumRecord]:
        """Return query and reference records embedded for a result row."""
        if self.matched_datasets is None:
            raise ValueError("This similarity result does not contain matched data")
        result_row = self.table.iloc[row]
        return (
            self.matched_datasets[0][int(result_row["data_index1"])],
            self.matched_datasets[1][int(result_row["data_index2"])],
        )

    @classmethod
    def with_matched_data(
        cls,
        table: pd.DataFrame,
        metadata: dict[str, Any],
        first: MSDataset,
        second: MSDataset,
    ) -> SimilarityDataset:
        """Attach each uniquely matched input spectrum once and index it from the table."""
        attached = table.copy()
        datasets: list[MSDataset] = []
        source_indices: list[np.ndarray] = []
        for side, dataset in enumerate((first, second), start=1):
            index_column = f"index{side}"
            data_column = f"data_index{side}"
            indices = attached[index_column].to_numpy(dtype=np.int64)
            if indices.size and (indices.min() < 0 or indices.max() >= len(dataset)):
                raise ValueError(f"'{index_column}' contains an invalid source index")
            unique = pd.unique(indices).astype(np.int64, copy=False)
            lookup = {int(source): local for local, source in enumerate(unique)}
            attached[data_column] = np.fromiter(
                (lookup[int(source)] for source in indices),
                dtype=np.int64,
                count=len(indices),
            )
            datasets.append(dataset[unique].copy())
            source_indices.append(unique)
        return cls(
            attached,
            metadata,
            (datasets[0], datasets[1]),
            (source_indices[0], source_indices[1]),
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
        table, matched_datasets, matched_source_indices = self._compact_matched_data()
        payload = table.to_parquet(index=False)
        metadata = json.dumps(self.metadata, ensure_ascii=False, allow_nan=False)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(descriptor)
        try:
            with h5py.File(temporary, "w") as handle:
                handle.attrs["format"] = "msentity.similarity"
                handle.attrs["schema_version"] = 2
                handle.attrs["data_mode"] = (
                    "embedded" if matched_datasets is not None else "lightweight"
                )
                handle.create_dataset("table.parquet", data=np.frombuffer(payload, dtype=np.uint8))
                handle.create_dataset("metadata.json", data=metadata,
                                      dtype=h5py.string_dtype("utf-8"))
                if matched_datasets is not None and matched_source_indices is not None:
                    embedded = handle.create_group("matched_data")
                    for side, (dataset, source_indices) in enumerate(
                        zip(matched_datasets, matched_source_indices), start=1
                    ):
                        group = embedded.create_group(str(side))
                        group.create_dataset("source_indices", data=source_indices)
                        group.create_dataset(
                            "dataset.msds",
                            data=np.frombuffer(self._dataset_to_bytes(dataset), dtype=np.uint8),
                            compression="gzip",
                        )
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path) -> SimilarityDataset:
        """Load a version 1 or version 2 ``.mssim`` result."""
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("format") != "msentity.similarity":
                raise ValueError("Not an msentity similarity file")
            version = int(handle.attrs.get("schema_version", 0))
            if version not in {1, 2}:
                raise ValueError("Unsupported similarity schema version")
            table = pd.read_parquet(io.BytesIO(handle["table.parquet"][()].tobytes()))
            metadata = json.loads(handle["metadata.json"].asstr()[()])
            matched_datasets = None
            matched_source_indices = None
            if version == 2 and "matched_data" in handle:
                groups = [handle["matched_data"][str(side)] for side in (1, 2)]
                matched_datasets = (
                    cls._dataset_from_bytes(groups[0]["dataset.msds"][()].tobytes()),
                    cls._dataset_from_bytes(groups[1]["dataset.msds"][()].tobytes()),
                )
                matched_source_indices = (
                    np.asarray(groups[0]["source_indices"][:], dtype=np.int64),
                    np.asarray(groups[1]["source_indices"][:], dtype=np.int64),
                )
        return cls(table, metadata, matched_datasets, matched_source_indices)

    def _compact_matched_data(
        self,
    ) -> tuple[
        pd.DataFrame,
        tuple[MSDataset, MSDataset] | None,
        tuple[np.ndarray, np.ndarray] | None,
    ]:
        matched_datasets = self.matched_datasets
        matched_source_indices = self.matched_source_indices
        if matched_datasets is None or matched_source_indices is None:
            return self.table, None, None
        table = self.table.copy()
        datasets: list[MSDataset] = []
        source_indices: list[np.ndarray] = []
        for side in (0, 1):
            column = f"data_index{side + 1}"
            referenced = pd.unique(table[column].to_numpy(dtype=np.int64)).astype(
                np.int64, copy=False
            )
            lookup = {int(old): new for new, old in enumerate(referenced)}
            table[column] = np.fromiter(
                (lookup[int(old)] for old in table[column]),
                dtype=np.int64,
                count=len(table),
            )
            datasets.append(matched_datasets[side][referenced].copy())
            source_indices.append(matched_source_indices[side][referenced])
        return (
            table,
            (datasets[0], datasets[1]),
            (source_indices[0], source_indices[1]),
        )

    @staticmethod
    def _dataset_to_bytes(dataset: MSDataset) -> bytes:
        descriptor, path = tempfile.mkstemp(suffix=".msds")
        os.close(descriptor)
        try:
            dataset.save(path)
            return Path(path).read_bytes()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @staticmethod
    def _dataset_from_bytes(payload: bytes) -> MSDataset:
        descriptor, path = tempfile.mkstemp(suffix=".msds")
        os.close(descriptor)
        try:
            Path(path).write_bytes(payload)
            return MSDataset.load(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @classmethod
    def from_datasets(
        cls,
        ds1: MSDataset,
        ds2: MSDataset,
        **kwargs: Any,
    ) -> SimilarityDataset:
        """Calculate a similarity dataset from two mass-spectrum datasets."""
        return calculate_similarity(ds1, ds2, **kwargs)


def _source_metadata(dataset: MSDataset, path: str, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": path,
        "n_rows": dataset.n_rows,
        "description": dataset.description,
        "attributes": dataset.attributes,
        "tags": dataset.tags,
    }


def calculate_similarity(
    ds1: MSDataset,
    ds2: MSDataset,
    *,
    source1: str = "",
    source2: str = "",
    key1: str = "SpecID",
    key2: str = "SpecID",
    method: str = "cosine",
    bin_width: float = 0.01,
    intensity_exponent: float = 1.0,
    max_cum_peaks: int = 200_000,
    include_matched_data: bool = False,
) -> SimilarityDataset:
    """Compare records with equal unique metadata keys.

    Set ``include_matched_data`` to attach each matched record once and add
    compact ``data_index1`` and ``data_index2`` references to the result table.
    """
    parameters = dict(
        key1=key1,
        key2=key2,
        method=method,
        bin_width=bin_width,
        intensity_exponent=intensity_exponent,
        max_cum_peaks=max_cum_peaks,
    )
    table = similarity_by_key(
        ds1,
        ds2,
        key1=key1,
        key2=key2,
        method=method,
        bin_width=bin_width,
        intensity_exponent=intensity_exponent,
        max_cum_peaks=max_cum_peaks,
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "by_key",
        "algorithm": f"binned_{method}_similarity_by_key",
        "parameters": parameters,
        "matching": "unique_intersection",
        "sources": [
            _source_metadata(ds1, source1, "first"),
            _source_metadata(ds2, source2, "second"),
        ],
        "row_count": len(table), "index_basis": "zero-based input dataset position",
        "input_scope": "entire in-memory datasets (including unsaved edits)",
        "matched_data": "embedded" if include_matched_data else "lightweight",
    }
    if include_matched_data:
        return SimilarityDataset.with_matched_data(table, metadata, ds1, ds2)
    return SimilarityDataset(table, metadata)


def calculate_library_search(
    query: MSDataset,
    reference: MSDataset,
    *,
    query_source: str = "",
    reference_source: str = "",
    threshold: float = 0.8,
    method: str = "cosine",
    bin_width: float = 0.01,
    intensity_exponent: float = 1.0,
    max_pairs_per_call: int = 2_000_000,
    max_cum_peaks: int = 200_000,
    include_matched_data: bool = True,
    show_progress: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimilarityDataset:
    """Run an exhaustive, thresholded search against a reference library.

    Candidate pairs are processed in bounded chunks. By default, every unique
    record referenced by a retained match is embedded once in the result.
    """
    parameters = dict(
        threshold=threshold,
        method=method,
        bin_width=bin_width,
        intensity_exponent=intensity_exponent,
        max_pairs_per_call=max_pairs_per_call,
        max_cum_peaks=max_cum_peaks,
    )
    table = library_search(
        query,
        reference,
        threshold=threshold,
        method=method,
        bin_width=bin_width,
        intensity_exponent=intensity_exponent,
        max_pairs_per_call=max_pairs_per_call,
        max_cum_peaks=max_cum_peaks,
        show_progress=show_progress,
        progress_callback=progress_callback,
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "library_search",
        "algorithm": f"binned_{method}_similarity_library_search",
        "parameters": parameters,
        "matching": "all_pairs_at_or_above_threshold",
        "sources": [
            _source_metadata(query, query_source, "query"),
            _source_metadata(reference, reference_source, "reference"),
        ],
        "row_count": len(table),
        "candidate_pair_count": query.n_rows * reference.n_rows,
        "index_basis": "zero-based input dataset position",
        "input_scope": "entire in-memory datasets (including unsaved edits)",
        "matched_data": "embedded" if include_matched_data else "lightweight",
    }
    if include_matched_data:
        return SimilarityDataset.with_matched_data(table, metadata, query, reference)
    return SimilarityDataset(table, metadata)


# Compatibility alias for callers that used the initial result class name.
SimilarityResult = SimilarityDataset
