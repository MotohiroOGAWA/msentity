from __future__ import annotations

import argparse
import csv
import inspect
import io
import json
import math
import sys
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Any

PREFIX = "MSENTITY_JSON:"


def emit(message: dict[str, Any]) -> None:
    sys.stdout.write(PREFIX + json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n")
    sys.stdout.flush()


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    return str(value)


def apply_view(dataset: Any, filters: Any = None, sort: Any = None, columns: Any = None) -> Any:
    """Return a dataset view with UI filtering, row sorting, and column order applied."""
    import pandas as pd

    view = dataset[:]
    available = dataset.columns
    for condition in filters if isinstance(filters, list) else []:
        column = str(condition.get("column", ""))
        operator = str(condition.get("operator", "text_eq"))
        raw_value = condition.get("value", "")
        if column not in available or raw_value is None or str(raw_value) == "":
            continue

        series = view[column]
        if operator in {">", ">=", "<", "<="}:
            left = pd.to_numeric(series, errors="coerce")
            try:
                right = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Filter value for {column} must be numeric") from exc
            mask = left.notna() & {
                ">": left > right,
                ">=": left >= right,
                "<": left < right,
                "<=": left <= right,
            }[operator]
        elif operator == "numeric_eq":
            try:
                right = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Filter value for {column} must be numeric") from exc
            left = pd.to_numeric(series, errors="coerce")
            mask = left.notna() & (left == right)
        else:
            left = series.fillna("").astype(str).str.casefold()
            right = str(raw_value).casefold()
            if operator == "contains":
                mask = left.str.contains(right, regex=False)
            elif operator == "!=":
                mask = left != right
            else:
                mask = left == right
        view = view[mask]

    sort_items = [sort] if isinstance(sort, dict) else sort if isinstance(sort, list) else []
    sort_columns: list[str] = []
    sort_ascending: list[bool] = []
    for item in sort_items:
        if not isinstance(item, dict):
            continue
        sort_column = str(item.get("column", ""))
        if sort_column in available and sort_column not in sort_columns:
            sort_columns.append(sort_column)
            sort_ascending.append(str(item.get("direction", "asc")) != "desc")
    if sort_columns:
        sorted_metadata = view.metadata
        # Apply lower-priority keys first. Stable mergesort then preserves each
        # lower-priority ordering inside equal groups of every higher key.
        for sort_column, ascending in reversed(list(zip(sort_columns, sort_ascending))):
            sorted_metadata = sorted_metadata.sort_values(
                by=sort_column,
                ascending=ascending,
                kind="mergesort",
                na_position="last",
            )
        order = sorted_metadata.index.to_numpy()
        view = view[order]

    if isinstance(columns, list):
        ordered = [str(column) for column in columns if str(column) in available]
        view.columns = ordered
    return view


def serialize_spectrum(spectrum: Any) -> dict[str, Any]:
    metadata = spectrum.metadata
    return {
        "mz": json_value(spectrum.mz.tolist()),
        "intensity": json_value(spectrum.intensity.tolist()),
        "metadata_columns": [] if metadata is None else list(metadata.columns),
        "metadata": [] if metadata is None else [
            {str(k): json_value(v) for k, v in row.items()}
            for row in metadata.to_dict(orient="records")
        ],
    }


def serialize_page(dataset: Any, page: int, page_size: int, dataset_id: str, input_file: Path) -> dict[str, Any]:
    total_rows = len(dataset)
    total_pages = max(1, math.ceil(total_rows / page_size))
    page = min(max(0, int(page)), total_pages - 1)
    start = page * page_size
    stop = min(start + page_size, total_rows)

    page_view = dataset[start:stop]
    metadata = page_view.metadata.reset_index(drop=True)
    rows = [
        {str(key): json_value(cell) for key, cell in row.items()}
        for row in metadata.to_dict(orient="records")
    ]
    spectra = [
        {
            "index": start + local_index,
            **serialize_spectrum(spectrum),
        }
        for local_index, spectrum in enumerate(page_view.peaks)
    ]
    return {
        "dataset_id": dataset_id,
        "dataset_name": input_file.name,
        "dataset_path": str(input_file),
        "columns": [str(column) for column in metadata.columns],
        "rows": rows,
        "row_ids": [int(i) for i in page_view.peaks._index],
        "spectra": spectra,
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "row_offset": start,
        "description": str(getattr(dataset, "description", "") or ""),
        "attributes": json_value(getattr(dataset, "attributes", {})),
        "tags": json_value(getattr(dataset, "tags", [])),
    }


def read_delimited_compat(input_file: Path, file_type: str) -> Any:
    """Read the viewer TSV/CSV format when the installed msentity predates the format’s I/O."""
    import numpy as np
    import pandas as pd
    from msentity import MSDataset, PeakSeries

    text = input_file.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t" if file_type == "tsv" else ",")
    if reader.fieldnames is None or "Peak" not in reader.fieldnames:
        raise ValueError(f"{file_type.upper()} must contain a 'Peak' column")
    columns = [column for column in reader.fieldnames if column != "Peak"]
    metadata_rows: list[dict[str, Any]] = []
    peak_rows: list[tuple[float, float]] = []
    offsets = [0]
    for row_number, row in enumerate(reader, start=2):
        metadata_rows.append({column: row.get(column, "") or "" for column in columns})
        peak_text = row.get("Peak", "") or ""
        for peak_number, item in enumerate(peak_text.split(";"), start=1):
            if not item.strip():
                continue
            parts = [part.strip() for part in item.split(",")]
            if len(parts) != 2:
                raise ValueError(f"Invalid Peak at {file_type.upper()} row {row_number}, peak {peak_number}")
            peak_rows.append((float(parts[0]), float(parts[1])))
        offsets.append(len(peak_rows))
    metadata = pd.DataFrame(metadata_rows, columns=columns)
    for column in columns:
        numeric = pd.to_numeric(metadata[column], errors="coerce")
        nonempty = metadata[column].ne("")
        if numeric[nonempty].notna().all():
            metadata[column] = numeric
    peak_data = np.asarray(peak_rows, dtype=float).reshape((-1, 2))
    return MSDataset(metadata, PeakSeries(peak_data, np.asarray(offsets, dtype=np.int64)))


def write_delimited_compat(dataset: Any, output_file: Path, file_type: str) -> None:
    """Write the viewer TSV/CSV format without requiring a new msentity install."""
    with output_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t" if file_type == "tsv" else ",", lineterminator="\n")
        writer.writerow([*dataset.columns, "Peak"])
        for record in dataset:
            peak_text = ";".join(
                f"{format(float(peak.mz), '.17g')},{format(float(peak.intensity), '.17g')}"
                for peak in record.peaks
            )
            writer.writerow([*(record[column] for column in dataset.columns), peak_text])


def load_dataset(input_file: Path, file_type: str | None = None) -> Any:
    try:
        from msentity import load_ms_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Python package 'msentity' is not installed in this environment. "
            "Install msentity in the same Python environment used by the VS Code extension."
        ) from exc
    detected_type = (file_type or input_file.suffix.lstrip(".")).lower()

    if detected_type in {"tsv", "csv"}:
        try:
            import msentity
            reader = getattr(msentity, f"read_{detected_type}")
        except (ImportError, AttributeError):
            return read_delimited_compat(input_file, detected_type)
        return reader(input_file, show_progress=False)

    def progress(processed: int, total: int, success: int, records: int) -> None:
        emit({
            "type": "loading-progress",
            "file_type": detected_type,
            "processed": processed,
            "total": total,
            "success": success,
            "records": records,
            "percent": min(100.0, processed / total * 100.0) if total else 0.0,
        })

    if detected_type in {"msp", "mgf"}:
        emit({"type": "loading-progress", "file_type": detected_type, "processed": 0, "total": input_file.stat().st_size, "success": 0, "records": 0, "percent": 0.0})
        # Keep the extension compatible with msentity versions released before
        # progress_callback/show_progress were added to load_ms_dataset.
        parameters = inspect.signature(load_ms_dataset).parameters
        kwargs: dict[str, Any] = {}
        if "show_progress" in parameters:
            kwargs["show_progress"] = False
        if "progress_callback" in parameters:
            kwargs["progress_callback"] = progress
        if file_type is not None and "file_type" in parameters:
            kwargs["file_type"] = file_type
        dataset = load_ms_dataset(str(input_file), **kwargs)
        if "progress_callback" not in parameters:
            emit({"type": "loading-progress", "file_type": detected_type, "processed": input_file.stat().st_size, "total": input_file.stat().st_size, "success": len(dataset), "records": len(dataset), "percent": 100.0})
        return dataset
    # MSDS/HDF5 loading is not line based and needs no progress-only keyword
    # arguments. Calling the stable one-argument API also supports older builds.
    parameters = inspect.signature(load_ms_dataset).parameters
    kwargs = {"file_type": file_type} if file_type is not None and "file_type" in parameters else {}
    return load_ms_dataset(str(input_file), **kwargs)


def export_dataset(dataset: Any, output_file: Path, file_type: str | None = None) -> None:
    output_type = (file_type or output_file.suffix.lstrip(".")).lower()
    if output_type not in {"msds", "msp", "mgf", "tsv", "csv"}:
        raise ValueError("Output format must be msds, msp, mgf, tsv, or csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_type == "msds":
        dataset.save(str(output_file))
    elif output_type == "msp":
        from msentity import write_msp
        write_msp(dataset, str(output_file), show_progress=False)
    elif output_type == "mgf":
        from msentity import write_mgf
        write_mgf(dataset, str(output_file), show_progress=False)
    elif output_type in {"tsv", "csv"}:
        try:
            import msentity
            writer = getattr(msentity, f"write_{output_type}")
        except (ImportError, AttributeError):
            write_delimited_compat(dataset, output_file, output_type)
        else:
            writer(dataset, str(output_file), show_progress=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--file-type", choices=("msds", "msp", "mgf", "tsv", "csv"))
    args = parser.parse_args()

    input_file = args.input_file.expanduser().resolve()
    if not input_file.is_file():
        emit({"type": "error", "title": "File not found", "message": str(input_file)})
        return 2

    page_size = min(500, max(1, args.page_size))
    try:
        dataset = load_dataset(input_file, args.file_type)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        emit({
            "type": "error",
            "title": "Could not load dataset",
            "message": str(exc),
        })
        return 1

    initial_id = str(input_file)
    datasets: dict[str, dict[str, Any]] = {
        initial_id: {"dataset": dataset, "path": input_file, "file_type": args.file_type}
    }
    emit({
        "type": "backend-ready",
        "file": str(input_file),
        "total_rows": len(dataset),
        "page_size": page_size,
        "dataset": {"id": initial_id, "name": input_file.name, "path": str(input_file), "total_rows": len(dataset)},
    })

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_type = request.get("type")
            dataset_id = str(request.get("dataset_id") or initial_id)
            entry = datasets.get(dataset_id)
            if request_type == "page":
                if entry is None:
                    raise ValueError(f"Unknown dataset: {dataset_id}")
                page = int(request.get("page", 0))
                view = apply_view(entry["dataset"], request.get("filters"), request.get("sort"))
                payload = serialize_page(view, page, page_size, dataset_id, entry["path"])
                payload["all_columns"] = entry["dataset"].columns
                emit({"type": "dataset-page", "value": payload})
            elif request_type in {"update-metadata", "update-cell", "remove-dataset", "add-column", "add-peak-column", "update-peak-cell", "get-peak-record"}:
                try:
                    if entry is None:
                        raise ValueError(f"Unknown dataset: {dataset_id}")
                    current = entry["dataset"]
                    if request_type == "remove-dataset":
                        del datasets[dataset_id]
                        emit({"type": "dataset-removed", "dataset_id": dataset_id})
                        continue
                    if request_type in {"add-column", "add-peak-column"}:
                        column = request.get("column")
                        initial = request.get("value", "")
                        if not isinstance(column, str) or not column.strip():
                            raise ValueError("Column name must not be empty")
                        column = column.strip()
                        if not isinstance(initial, str):
                            raise ValueError("Initial value must be text")
                        peak_column = request_type == "add-peak-column"
                        target = current.peaks if peak_column else current
                        existing = current.peaks.metadata_columns if peak_column else current.columns
                        reserved = {"mz", "m/z", "intensity"} if peak_column else {"peak"}
                        if column in existing or column.casefold() in reserved:
                            raise ValueError("Column name already exists or is reserved")
                        target[column] = initial
                        emit({"type": "peak-columns-updated" if peak_column else "column-added",
                              "dataset_id": dataset_id, "column": column})
                        continue
                    if request_type in {"update-peak-cell", "get-peak-record"}:
                        row_id = request.get("row_id")
                        indices = current.peaks._index.tolist()
                        if type(row_id) is not int or row_id not in indices:
                            raise ValueError("Unknown spectrum")
                        spectrum = current.peaks[indices.index(row_id)]
                        if request_type == "update-peak-cell":
                            column = request.get("column")
                            peak_index = request.get("peak_index")
                            new_value = request.get("value")
                            if column not in current.peaks.metadata_columns:
                                raise ValueError("Unknown peak annotation column")
                            if type(peak_index) is not int or not 0 <= peak_index < len(spectrum):
                                raise ValueError("Unknown peak")
                            if not isinstance(new_value, str):
                                raise ValueError("Annotation must be text")
                            # Preserve the column's scalar type when editing existing annotations.
                            old = json_value(spectrum.metadata.iloc[peak_index][column])
                            if isinstance(old, bool):
                                if new_value not in {"true", "false"}:
                                    raise ValueError("Enter true or false")
                                new_value = new_value == "true"
                            elif isinstance(old, (int, float)):
                                new_value = int(new_value) if isinstance(old, int) and new_value.lstrip("+-").isdigit() else float(new_value)
                                if not math.isfinite(new_value):
                                    raise ValueError("Enter a finite number")
                            values = current.peaks._metadata_ref[column].astype(object).copy()
                            values.iloc[int(current.peaks._offsets_ref[row_id]) + peak_index] = new_value
                            current.peaks._metadata_ref[column] = values.infer_objects()
                        emit({"type": "peak-record", "dataset_id": dataset_id, "row_id": row_id,
                              "modified": request_type == "update-peak-cell",
                              "spectrum": serialize_spectrum(spectrum)})
                        continue
                    if request_type == "update-metadata":
                        description = request.get("description")
                        attributes = request.get("attributes")
                        tags = request.get("tags")
                        if not isinstance(description, str):
                            raise ValueError("Description must be text")
                        if not isinstance(attributes, dict) or any(
                            not isinstance(k, str) or not isinstance(v, str) for k, v in attributes.items()
                        ):
                            raise ValueError("Attribute names and values must be text")
                        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                            raise ValueError("Tags must be a list of text values")
                        current.description = description
                        current.attributes = attributes
                        current.tags = tags
                    else:
                        column = request.get("column")
                        row_id = request.get("row_id")
                        if column not in current.columns:
                            raise ValueError("Unknown metadata column")
                        indices = current.peaks._index.tolist()
                        if type(row_id) is not int or row_id not in indices:
                            raise ValueError("Unknown spectrum")
                        new_value = request.get("value")
                        old_value = json_value(current[indices.index(row_id)][column])
                        if isinstance(old_value, bool):
                            if new_value not in ("true", "false"):
                                raise ValueError("Enter true or false")
                            new_value = new_value == "true"
                        elif isinstance(old_value, (int, float)):
                            new_value = int(new_value) if isinstance(old_value, int) and str(new_value).lstrip("+-").isdigit() else float(new_value)
                            if not math.isfinite(new_value):
                                raise ValueError("Enter a finite number")
                        elif old_value is not None and not isinstance(old_value, str):
                            raise ValueError("Only text, numeric and boolean cells can be edited")
                        # Replace the backing column to allow integer-to-float edits
                        # under pandas versions that reject incompatible .loc writes.
                        values = current._spectrum_metadata_ref[column].astype(object).copy()
                        values.iloc[row_id] = new_value
                        current._spectrum_metadata_ref[column] = values.infer_objects()
                    response = {"type": "metadata-updated", "dataset_id": dataset_id}
                    if request_type == "update-metadata":
                        response["metadata"] = {
                            "description": current.description,
                            "attributes": json_value(current.attributes),
                            "tags": json_value(current.tags),
                        }
                    emit(response)
                except Exception as exc:
                    emit({"type": "edit-error", "dataset_id": dataset_id, "message": str(exc)})
            elif request_type == "add-dataset":
                added_path = Path(str(request.get("path", ""))).expanduser().resolve()
                if not added_path.is_file():
                    raise FileNotFoundError(str(added_path))
                added_id = str(added_path)
                if added_id not in datasets:
                    added_type = request.get("file_type")
                    added_dataset = load_dataset(added_path, added_type)
                    datasets[added_id] = {"dataset": added_dataset, "path": added_path, "file_type": added_type}
                added_entry = datasets[added_id]
                emit({"type": "dataset-added", "dataset": {"id": added_id, "name": added_path.name, "path": added_id, "total_rows": len(added_entry["dataset"])}})
                emit({"type": "dataset-page", "value": serialize_page(added_entry["dataset"], 0, page_size, added_id, added_path)})
            elif request_type == "reload":
                if entry is None:
                    raise ValueError(f"Unknown dataset: {dataset_id}")
                entry["dataset"] = load_dataset(entry["path"], entry["file_type"])
                emit({"type": "dataset-reloaded", "dataset_id": dataset_id})
                view = apply_view(entry["dataset"], request.get("filters"), request.get("sort"))
                payload = serialize_page(view, 0, page_size, dataset_id, entry["path"])
                payload["all_columns"] = entry["dataset"].columns
                emit({"type": "dataset-page", "value": payload})
            elif request_type == "similarity-options":
                emit({"type": "similarity-options", "datasets": [
                    {"id": identifier, "name": item["path"].name,
                     "columns": item["dataset"].columns}
                    for identifier, item in datasets.items()
                ], "active_dataset_id": dataset_id})
            elif request_type == "calculate-similarity":
                try:
                    from msentity.similarity import (
                        calculate_library_search,
                        calculate_similarity,
                    )

                    mode = str(request.get("mode", "by_key"))
                    id1 = str(request["dataset1"])
                    if id1 not in datasets:
                        raise ValueError("Select a loaded query dataset")
                    first = datasets[id1]
                    reference_path = request.get("reference_path")
                    if reference_path:
                        second_path = Path(str(reference_path)).expanduser().resolve()
                        if not second_path.is_file():
                            raise FileNotFoundError(str(second_path))
                        second = {
                            "dataset": load_dataset(second_path),
                            "path": second_path,
                            "file_type": None,
                        }
                    else:
                        id2 = str(request.get("dataset2", ""))
                        if id2 not in datasets:
                            raise ValueError("Select a loaded reference dataset or a reference file")
                        second = datasets[id2]
                    target = Path(request["path"]).expanduser().resolve()
                    input_paths = {item["path"] for item in datasets.values()} | {second["path"]}
                    if target in input_paths:
                        raise ValueError("Output must not overwrite an input dataset")
                    parameters = request.get("parameters", {})
                    if not isinstance(parameters, dict):
                        raise ValueError("Similarity parameters must be an object")
                    if mode == "library_search":
                        def similarity_progress(processed: int, total: int) -> None:
                            emit({
                                "type": "similarity-progress",
                                "processed": processed,
                                "total": total,
                                "percent": processed / total * 100.0 if total else 100.0,
                            })

                        result = calculate_library_search(
                            first["dataset"], second["dataset"],
                            query_source=str(first["path"]),
                            reference_source=str(second["path"]),
                            progress_callback=similarity_progress,
                            **parameters,
                        )
                    elif mode == "by_key":
                        if first["path"] == second["path"]:
                            raise ValueError("Select two different datasets for metadata-key matching")
                        result = calculate_similarity(
                            first["dataset"], second["dataset"],
                            source1=str(first["path"]), source2=str(second["path"]),
                            **parameters,
                        )
                    else:
                        raise ValueError(f"Unknown similarity mode: {mode}")
                    result.save(target)
                    emit({"type": "similarity-complete", "path": str(target),
                          "total_rows": len(result.table),
                          "candidate_pairs": result.metadata.get("candidate_pair_count")})
                except Exception as exc:
                    traceback.print_exc(file=sys.stderr)
                    emit({"type": "similarity-error", "message": str(exc)})
            elif request_type == "assign-spec-id":
                try:
                    from msentity.processing.id import set_spec_id

                    if entry is None:
                        raise ValueError(f"Unknown dataset: {dataset_id}")
                    prefix = request.get("prefix", "")
                    overwrite = request.get("overwrite", False)
                    if not isinstance(prefix, str) or not isinstance(overwrite, bool):
                        raise ValueError("prefix must be a string and overwrite must be a boolean")
                    changed = set_spec_id(entry["dataset"], prefix=prefix, overwrite=overwrite)
                    if not changed:
                        raise ValueError("SpecID already exists. Allow replacement to assign new IDs.")
                    emit({"type": "spec-id-complete", "dataset_id": dataset_id, "total_rows": len(entry["dataset"])})
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc(file=sys.stderr)
                    emit({"type": "spec-id-error", "dataset_id": dataset_id, "message": str(exc)})
            elif request_type == "export":
                if entry is None:
                    raise ValueError(f"Unknown dataset: {dataset_id}")
                output_file = Path(str(request.get("path", ""))).expanduser().resolve()
                emit({"type": "export-start", "path": str(output_file)})
                try:
                    view = apply_view(entry["dataset"], request.get("filters"), request.get("sort"), request.get("columns"))
                    export_dataset(view, output_file, request.get("file_type"))
                    emit({"type": "export-complete", "path": str(output_file), "total_rows": len(view)})
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc(file=sys.stderr)
                    emit({"type": "export-error", "message": str(exc)})
            else:
                emit({"type": "error", "title": "Unknown request", "message": str(request_type)})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            emit({"type": "error", "title": "Backend error", "message": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
