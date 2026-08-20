from __future__ import annotations

import argparse
import inspect
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


def serialize_page(dataset: Any, page: int, page_size: int) -> dict[str, Any]:
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


def load_dataset(input_file: Path, file_type: str | None = None) -> Any:
    try:
        from msentity import load_ms_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Python package 'msentity' is not installed in this environment. "
            "Install msentity in the same Python environment used by the VS Code extension."
        ) from exc
    detected_type = (file_type or input_file.suffix.lstrip(".")).lower()

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
    if output_type not in {"msds", "msp", "mgf"}:
        raise ValueError("Output format must be msds, msp, or mgf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_type == "msds":
        dataset.save(str(output_file))
    elif output_type == "msp":
        from msentity import write_msp
        write_msp(dataset, str(output_file), show_progress=False)
    elif output_type == "mgf":
        from msentity import write_mgf
        write_mgf(dataset, str(output_file), show_progress=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--file-type", choices=("msds", "msp", "mgf"))
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

    emit({
        "type": "backend-ready",
        "file": str(input_file),
        "total_rows": len(dataset),
        "page_size": page_size,
    })

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_type = request.get("type")
            if request_type == "page":
                page = int(request.get("page", 0))
                emit({"type": "dataset-page", "value": serialize_page(dataset, page, page_size)})
            elif request_type == "reload":
                dataset = load_dataset(input_file, args.file_type)
                emit({"type": "dataset-page", "value": serialize_page(dataset, 0, page_size)})
            elif request_type == "export":
                output_file = Path(str(request.get("path", ""))).expanduser().resolve()
                emit({"type": "export-start", "path": str(output_file)})
                try:
                    export_dataset(dataset, output_file, request.get("file_type"))
                    emit({"type": "export-complete", "path": str(output_file), "total_rows": len(dataset)})
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
