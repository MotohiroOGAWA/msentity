from __future__ import annotations

import argparse
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


def load_dataset(input_file: Path) -> Any:
    try:
        from msentity import load_ms_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Python package 'msentity' is not installed in this environment. "
            "Install msentity in the same Python environment used by the VS Code extension."
        ) from exc
    return load_ms_dataset(str(input_file))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--page-size", type=int, default=20)
    args = parser.parse_args()

    input_file = args.input_file.expanduser().resolve()
    if not input_file.is_file():
        emit({"type": "error", "title": "File not found", "message": str(input_file)})
        return 2

    page_size = min(500, max(1, args.page_size))
    try:
        dataset = load_dataset(input_file)
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
                dataset = load_dataset(input_file)
                emit({"type": "dataset-page", "value": serialize_page(dataset, 0, page_size)})
            else:
                emit({"type": "error", "title": "Unknown request", "message": str(request_type)})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            emit({"type": "error", "title": "Backend error", "message": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
