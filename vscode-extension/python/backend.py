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
            "mz": json_value(spectrum.mz),
            "intensity": json_value(spectrum.intensity),
        }
        for local_index, spectrum in enumerate(page_view.peaks)
    ]
    return {
        "dataset_id": dataset_id,
        "dataset_name": input_file.name,
        "dataset_path": str(input_file),
        "columns": [str(column) for column in metadata.columns],
        "rows": rows,
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
                view = apply_view(entry["dataset"], request.get("filters"), request.get("sort"))
                payload = serialize_page(view, 0, page_size, dataset_id, entry["path"])
                payload["all_columns"] = entry["dataset"].columns
                emit({"type": "dataset-page", "value": payload})
            elif request_type == "similarity-options":
                emit({"type": "similarity-options", "datasets": [
                    {"id": identifier, "name": item["path"].name,
                     "columns": item["dataset"].columns}
                    for identifier, item in datasets.items()
                ]})
            elif request_type == "calculate-similarity":
                try:
                    from msentity.similarity import SimilarityDataset

                    id1, id2 = request["dataset1"], request["dataset2"]
                    if id1 == id2 or id1 not in datasets or id2 not in datasets:
                        raise ValueError("Select two different loaded datasets")
                    first, second = datasets[id1], datasets[id2]
                    target = Path(request["path"]).expanduser().resolve()
                    if target in {item["path"] for item in datasets.values()}:
                        raise ValueError("Output must not overwrite an input dataset")
                    result = SimilarityDataset.from_datasets(
                        first["dataset"], second["dataset"],
                        source1=str(first["path"]), source2=str(second["path"]),
                        **request.get("parameters", {}),
                    )
                    result.save(target)
                    emit({"type": "similarity-complete", "path": str(target),
                          "total_rows": len(result.table)})
                except Exception as exc:
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
