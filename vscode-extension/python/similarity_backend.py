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
    page_view = view.iloc[page * page_size:(page + 1) * page_size]
    return json_value({
        "columns": list(view.columns),
        "rows": page_view.to_dict(orient="records"),
        "result_indices": page_view.index.to_list(),
        "page": page, "total_pages": pages, "page_size": page_size,
        "total_rows": len(view), "unfiltered_rows": len(result.table),
        "metadata": result.metadata,
        "has_matched_data": result.has_matched_data,
        "matched_data_rows": [len(dataset) for dataset in result.matched_datasets]
        if result.matched_datasets is not None else [0, 0],
        "histogram": {"counts": counts, "edges": edges},
        "statistics": statistics,
    })


def match_payload(result, row):
    if not result.has_matched_data:
        raise ValueError("This similarity result does not contain matched data")
    result_row = result.table.iloc[row]
    sources = result.metadata.get("sources", [{}, {}])
    if not isinstance(sources, list):
        sources = [{}, {}]
    payloads = []
    for side in (0, 1):
        record = result.matched_datasets[side][int(result_row[f"data_index{side + 1}"])]
        source = sources[side] if side < len(sources) and isinstance(sources[side], dict) else {}
        source_index = int(result.matched_source_indices[side][int(result_row[f"data_index{side + 1}"])])
        metadata = {column: json_value(record[column]) for column in record.columns}
        title = next((str(metadata[key]) for key in
                      ("Name", "name", "TITLE", "Title", "SpecID", "ID", "id")
                      if metadata.get(key) not in (None, "")), f"Spectrum {source_index + 1}")
        source_path = str(source.get("path", ""))
        payloads.append({
            "datasetName": Path(source_path).name or str(source.get("role", f"Dataset {side + 1}")),
            "datasetPath": source_path,
            "globalIndex": source_index,
            "title": title,
            "columns": record.columns,
            "row": metadata,
            "spectrum": {
                "mz": json_value(record.spectrum.mz),
                "intensity": json_value(record.spectrum.intensity),
            },
        })
    return {
        "query": payloads[0],
        "reference": payloads[1],
        "method": result.metadata.get("parameters", {}).get("method", "cosine"),
    }


def export_result(result, request):
    view = filtered_table(result.table, request.get("filters"), request.get("sort"))
    path = Path(request["path"])
    suffix = path.suffix.lower()
    if suffix == ".mssim":
        metadata = dict(result.metadata, row_count=len(view), export_view={
            "filters": request.get("filters", []), "sort": request.get("sort"),
            "source_rows": len(result.table),
        })
        SimilarityDataset(
            view.reset_index(drop=True),
            metadata,
            result.matched_datasets,
            result.matched_source_indices,
        ).save(path)
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
            elif kind == "match":
                emit({"type": "similarity-match", **match_payload(result, int(request["row"]))})
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
