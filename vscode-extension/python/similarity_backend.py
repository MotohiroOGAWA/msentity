"""JSON protocol for the dedicated similarity result editor."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from backend import emit, json_value
from msentity.similarity import SimilarityDataset


def filtered_table(table, filters=None, sort=None):
    view = table
    for condition in filters or []:
        column = condition["column"]
        if column not in view.columns:
            raise ValueError(f"Unknown filter column: {column}")
        operator = condition["operator"]
        raw = condition.get("value", "")
        if operator in {">", ">=", "<", "<=", "numeric_eq"}:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("Filter value must be finite")
            series = pd.to_numeric(view[column], errors="coerce")
            masks = {">": series > value, ">=": series >= value,
                     "<": series < value, "<=": series <= value, "numeric_eq": series == value}
            mask = masks[operator]
        else:
            series = view[column].fillna("").astype(str)
            if operator == "contains":
                mask = series.str.contains(str(raw), regex=False, case=False)
            elif operator == "text_eq":
                mask = series == str(raw)
            elif operator == "!=":
                mask = series != str(raw)
            else:
                raise ValueError(f"Unknown filter operator: {operator}")
        view = view[mask]
    if sort:
        view = view.sort_values(sort["column"], ascending=sort.get("direction") != "desc",
                                kind="mergesort")
    return view


def page_payload(result, request, page_size):
    view = filtered_table(result.table, request.get("filters"), request.get("sort"))
    bins = int(request.get("bins", 20))
    if not 1 <= bins <= 200:
        raise ValueError("Histogram bins must be between 1 and 200")
    view_result = SimilarityDataset(view.reset_index(drop=True), result.metadata)
    counts, edges = view_result.histogram(bins)
    statistics = view_result.describe_scores()
    statistics.pop("count")
    pages = max(1, math.ceil(len(view) / page_size))
    page = min(max(0, int(request.get("page", 0))), pages - 1)
    return json_value({
        "columns": list(view.columns),
        "rows": view.iloc[page * page_size:(page + 1) * page_size].to_dict(orient="records"),
        "page": page, "total_pages": pages, "page_size": page_size,
        "total_rows": len(view), "unfiltered_rows": len(result.table),
        "metadata": result.metadata,
        "histogram": {"counts": counts, "edges": edges},
        "statistics": statistics,
    })


def export_result(result, request):
    view = filtered_table(result.table, request.get("filters"), request.get("sort"))
    path = Path(request["path"])
    suffix = path.suffix.lower()
    if suffix == ".mssim":
        metadata = dict(result.metadata, row_count=len(view), export_view={
            "filters": request.get("filters", []), "sort": request.get("sort"),
            "source_rows": len(result.table),
        })
        SimilarityDataset(view, metadata).save(path)
    elif suffix == ".parquet":
        view.to_parquet(path, index=False)
    elif suffix in {".csv", ".tsv"}:
        view.to_csv(path, index=False, sep="\t" if suffix == ".tsv" else ",")
    else:
        raise ValueError("Export format must be mssim, parquet, csv, or tsv")
    return len(view)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--page-size", type=int, default=50)
    args = parser.parse_args()
    try:
        result = SimilarityDataset.load(args.input_file)
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        return 1
    emit({"type": "backend-ready"})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            kind = request.get("type")
            if kind == "reload":
                result = SimilarityDataset.load(args.input_file)
            if kind in {"page", "reload"}:
                emit({"type": "similarity-page", "value": page_payload(result, request, args.page_size)})
            elif kind == "export":
                count = export_result(result, request)
                emit({"type": "export-complete", "path": request["path"], "total_rows": count})
            else:
                raise ValueError(f"Unknown request: {kind}")
        except Exception as exc:
            emit({"type": "error", "message": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
