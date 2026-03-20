from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from tqdm import tqdm

from .IOContext import (
    ReaderContext,
    is_peak_header_line,
    parse_peak_line,
    _quote_peak_meta_item,
    _stringify_msp_value,
)
from .constants import ErrorLogLevel
from ..core.MSDataset import MSDataset
from ..processing.id import set_spec_id


def read_mgf(
    filepath: str,
    *,
    encoding: str = "utf-8",
    return_header_map: bool = False,
    spec_id_prefix: Optional[str] = None,
    error_log_level: ErrorLogLevel = ErrorLogLevel.NONE,
    error_log_file: Optional[str] = None,
    allow_duplicate_cols: bool = False,
    show_progress: bool = True,
    canonicalize_column: bool = True,
    canonicalize_adduct_type: bool = True,
    error_context_lines: int = 10,
    peak_parser: Optional[Callable[[str], Dict[str, Any]]] = None,
    auto_peak_col_prefix: str = "column",
) -> MSDataset | tuple[MSDataset, dict[str, str]]:
    """
    Read an MGF file into an MSDataset.

    This implementation parses peak lines incrementally instead of buffering
    the entire peak text block for each record.

    Parameters
    ----------
    filepath
        Path to the MGF file.
    encoding
        File encoding.
    return_header_map
        Whether to also return the original header map.
    spec_id_prefix
        Prefix for generated spectrum IDs.
    error_log_level
        Error logging level.
    error_log_file
        Error log output file path.
    allow_duplicate_cols
        Whether duplicate metadata keys are allowed within one record.
    show_progress
        Whether to show a progress bar.
    canonicalize_column
        Whether metadata keys are canonicalized.
    canonicalize_adduct_type
        Whether adduct type values are canonicalized.
    error_context_lines
        Number of recent lines kept for error context.
    peak_parser
        Optional custom parser for a single peak line.
        It must return a dictionary such as
        ``{"mz": ..., "intensity": ..., ...}``.
    auto_peak_col_prefix
        Prefix for automatically generated peak metadata columns.

    Returns
    -------
    MSDataset or tuple
        Parsed dataset, optionally with ``header_map``.
    """
    mgf_reader = ReaderContext(
        file_path=filepath,
        file_type_name="mgf",
        error_log_level=error_log_level,
        error_log_file=error_log_file,
        encoding=encoding,
        allow_duplicate_cols=allow_duplicate_cols,
        show_progress=show_progress,
        canonicalize_column=canonicalize_column,
        canonicalize_adduct_type=canonicalize_adduct_type,
        error_context_lines=error_context_lines,
    )

    in_ions_block = False
    peak_flag = False
    peak_columns: Optional[List[str]] = None

    with open(filepath, "r", encoding=encoding) as f:
        for line in f:
            mgf_reader.update(line)

            try:
                stripped = line.strip()
                upper = stripped.upper()

                if stripped == "":
                    continue

                if upper == "BEGIN IONS":
                    if in_ions_block and (mgf_reader.meta or len(mgf_reader.peak["mz"]) > 0):
                        mgf_reader.update_record()
                    in_ions_block = True
                    peak_flag = False
                    peak_columns = None
                    continue

                if upper == "END IONS":
                    if in_ions_block:
                        mgf_reader.update_record()
                    in_ions_block = False
                    peak_flag = False
                    peak_columns = None
                    continue

                if not in_ions_block:
                    continue

                if peak_flag:
                    if peak_columns is None and is_peak_header_line(line):
                        peak_columns = line.strip().split()
                        continue

                    try:
                        if peak_parser is None:
                            peak_entry = parse_peak_line(
                                line,
                                peak_columns=peak_columns,
                                auto_col_prefix=auto_peak_col_prefix,
                            )
                        else:
                            peak_entry = peak_parser(line)

                        mgf_reader.add_peak(**peak_entry)

                    except Exception as e:
                        mgf_reader.add_error_message(str(e), line_text=line)

                    continue

                # metadata line
                if "=" in line:
                    key, value = line.split("=", 1)
                    mgf_reader.add_meta(key, value)
                    continue

                # first line of peak section
                peak_flag = True

                if is_peak_header_line(line):
                    peak_columns = line.strip().split()
                    continue

                peak_columns = None

                try:
                    if peak_parser is None:
                        peak_entry = parse_peak_line(
                            line,
                            peak_columns=peak_columns,
                            auto_col_prefix=auto_peak_col_prefix,
                        )
                    else:
                        peak_entry = peak_parser(line)

                    mgf_reader.add_peak(**peak_entry)

                except Exception as e:
                    mgf_reader.add_error_message(str(e), line_text=line)

            except Exception as e:
                mgf_reader.add_error_message(str(e), line_text=line)

    if in_ions_block and (mgf_reader.meta or len(mgf_reader.peak["mz"]) > 0):
        mgf_reader.update_record()

    ms_dataset = mgf_reader.get_dataset()

    if spec_id_prefix is not None:
        set_spec_id(ms_dataset, prefix=spec_id_prefix)

    if return_header_map:
        return ms_dataset, mgf_reader.header_map
    return ms_dataset


def write_mgf(
    dataset: MSDataset,
    path: str,
    *,
    headers: Optional[Sequence[str]] = None,
    header_map: Optional[Mapping[str, str]] = None,
    peak_headers: Optional[Sequence[str]] = None,
    encoding: str = "utf-8",
    delimiter: str = "\t",
    show_progress: bool = True,
) -> None:
    """
    Write an MSDataset to an MGF file.

    Parameters
    ----------
    dataset
        Target dataset.
    path
        Output MGF file path.
    headers
        Spectrum metadata columns to write. If omitted, all visible dataset
        columns are written.
    header_map
        Optional mapping from dataset column names to MGF header names.
        If a column is not present in this mapping, its original name is used.
    peak_headers
        Peak metadata columns to write. If omitted, all available peak metadata
        columns are considered except ``"mz"`` and ``"intensity"``.
    encoding
        Output text encoding.
    delimiter
        Delimiter between m/z, intensity, and optional peak metadata.
    show_progress
        Whether to show a progress bar.

    Returns
    -------
    None
    """
    spectrum_df = dataset.metadata
    header_map_dict = dict(header_map) if header_map is not None else {}

    if headers is None:
        headers = dataset.columns

    valid_headers: list[str] = []
    for col in headers:
        if col not in spectrum_df.columns:
            continue
        if col == "NumPeaks":
            continue
        valid_headers.append(col)

    for col in valid_headers:
        if col not in header_map_dict:
            header_map_dict[col] = col

    peak_metadata_df = dataset.peaks.metadata
    if peak_headers is None:
        if peak_metadata_df is not None:
            peak_headers = peak_metadata_df.columns.tolist()
        else:
            peak_headers = []

    valid_peak_headers: list[str] = []
    if peak_metadata_df is not None:
        for col in peak_headers:
            if col not in peak_metadata_df.columns:
                continue
            if col in ("mz", "intensity"):
                continue
            valid_peak_headers.append(col)

    pbar = (
        tqdm(total=len(dataset), desc="[Writing MGF]", unit="record", mininterval=1.0)
        if show_progress
        else None
    )

    success_count = 0
    processed_count = 0

    try:
        with open(path, "w", encoding=encoding) as wf:
            for record in dataset:
                try:
                    wf.write("BEGIN IONS\n")

                    for key in valid_headers:
                        value = _stringify_msp_value(record[key])
                        wf.write(f"{header_map_dict[key]}={value}\n")

                    used_peak_columns: set[str] = set()
                    peak_rows: list[tuple[str, list[str]]] = []

                    for peak in record.peaks:
                        mz_intensity_text = f"{peak.mz}{delimiter}{peak.intensity}"

                        meta_items: list[str] = []
                        for col in valid_peak_headers:
                            item = _quote_peak_meta_item(peak.metadata.get(col, ""))
                            if item != "":
                                used_peak_columns.add(col)
                            meta_items.append(item)

                        peak_rows.append((mz_intensity_text, meta_items))

                    used_peak_headers = [col for col in valid_peak_headers if col in used_peak_columns]
                    used_peak_header_idxs = [
                        i for i, col in enumerate(valid_peak_headers) if col in used_peak_columns
                    ]

                    if used_peak_headers:
                        wf.write(delimiter.join(["mz", "intensity", *used_peak_headers]) + "\n")

                    for mz_intensity_text, meta_items in peak_rows:
                        selected_items = [meta_items[i] for i in used_peak_header_idxs]

                        if not selected_items or all(item == "" for item in selected_items):
                            wf.write(f"{mz_intensity_text}\n")
                        else:
                            meta_text = '" ; "'.join(selected_items)
                            wf.write(f'{mz_intensity_text}{delimiter}"{meta_text}"\n')

                    wf.write("END IONS\n\n")
                    success_count += 1

                except Exception as e:
                    print(f"Error writing record {processed_count}: {e}")

                finally:
                    processed_count += 1
                    if pbar is not None:
                        pbar.update(1)
                        ratio = (success_count / processed_count * 100.0) if processed_count > 0 else 0.0
                        pbar.set_postfix(
                            {"Success": f"{success_count}/{processed_count}({ratio:.1f}%)"}
                        )
    finally:
        if pbar is not None:
            pbar.close()