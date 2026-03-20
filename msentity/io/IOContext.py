from __future__ import annotations

import os
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .ItemParser import ItemParser
from .constants import ErrorLogLevel
from ..core.MSDataset import MSDataset
from ..core.PeakSeries import PeakSeries


class ReaderContext:
    """
    Context object for line-based spectrum file readers.
    """

    def __init__(
        self,
        file_path: str,
        *,
        file_type_name: str = "",
        error_log_level: ErrorLogLevel = ErrorLogLevel.NONE,
        error_log_file: Optional[str] = None,
        encoding: str = "utf-8",
        allow_duplicate_cols: bool = False,
        show_progress: bool = True,
        canonicalize_column: bool = True,
        canonicalize_adduct_type: bool = True,
        error_context_lines: int = 10,
    ) -> None:
        self.file_type_name = file_type_name
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.processed_size = 0
        self.encoding = encoding
        self.allow_duplicate_cols = allow_duplicate_cols
        self.item_parser = ItemParser(
            canonicalize_column=canonicalize_column,
            canonicalize_adduct_type=canonicalize_adduct_type,
        )
        self.error_log_level = error_log_level
        self.header_map: dict[str, str] = {}
        self.error_context_lines = error_context_lines

        self.all_cols: dict[str, list[Any]] = {}
        self.all_col_names: list[str] = []

        self.all_peak: dict[str, list[Any]] = {
            "mz": [],
            "intensity": [],
        }
        self.all_peak_meta_names: list[str] = []
        self.offsets: list[int] = [0]

        self.record_cnt = 0
        self.success_cnt = 0
        self.line_count = 0

        self._reset_record()
        self.error_file_path = self._resolve_error_log_path(error_log_file)
        self.pbar = self.progress_bar if show_progress else None

    def _resolve_error_log_path(self, error_log_file: Optional[str]) -> str:
        if error_log_file is None:
            now = datetime.now().strftime("%Y%m%d%H%M%S")
            base = os.path.splitext(self.file_path)[0] + f"_error_{now}"
            candidate = base + ".txt"

            count = 1
            while os.path.exists(candidate):
                candidate = f"{base}_{count}.txt"
                count += 1
            return candidate

        directory = os.path.dirname(error_log_file)
        if directory and not os.path.exists(directory):
            raise ValueError(f"Directory '{directory}' does not exist.")

        return error_log_file

    @property
    def progress_bar(self):
        name = os.path.basename(self.file_path)
        label = f"[Reading {self.file_type_name}]" if self.file_type_name else "[Reading]"
        return tqdm(total=self.file_size, desc=f"{label}{name}", mininterval=0.5)

    def update(self, line: str) -> None:
        """
        Update per-line state.
        """
        self.line_count += 1
        self.processed_size += len(line.encode(self.encoding))
        self.recent_lines.append(line.rstrip("\n"))

        if self.pbar is not None:
            self.pbar.update(len(line.encode(self.encoding)))

    def _reset_record(self) -> None:
        self.meta: dict[str, Any] = {}
        self.peak: dict[str, list[Any]] = {"mz": [], "intensity": []}
        self.error_text_list: list[str] = []
        self.error_flag = False
        self.recent_lines = deque(maxlen=self.error_context_lines)

    def _has_record_content(self) -> bool:
        return bool(self.meta) or (
            len(self.peak["mz"]) > 0 and len(self.peak["intensity"]) > 0
        )

    def _set_record(self) -> None:
        for key in self.meta:
            if key not in self.all_cols:
                self.all_cols[key] = [""] * self.success_cnt

        for key in self.all_cols:
            self.all_cols[key].append(self.meta.get(key, ""))

        for key in self.peak:
            if key not in self.all_peak:
                self.all_peak[key] = [""] * len(self.all_peak["mz"])
                self.all_peak_meta_names.append(key)

        peak_length = len(self.peak["mz"])
        for key in self.all_peak:
            self.all_peak[key].extend(self.peak.get(key, [""] * peak_length))

        self.offsets.append(len(self.all_peak["mz"]))

    def _cleanup_column_names(self) -> None:
        missing_meta_names = [k for k in self.all_col_names if k not in self.all_cols]
        for key in missing_meta_names:
            self.all_col_names.remove(key)

        missing_header_map = [k for k in self.header_map if k not in self.all_col_names]
        for key in missing_header_map:
            del self.header_map[key]

        missing_peak_meta = [k for k in self.all_peak_meta_names if k not in self.all_peak]
        for key in missing_peak_meta:
            self.all_peak_meta_names.remove(key)

    def _update_progress_postfix(self) -> None:
        if self.pbar is not None:
            ratio = (self.success_cnt / self.record_cnt * 100.0) if self.record_cnt > 0 else 0.0
            self.pbar.set_postfix_str(
                f"Success:{self.success_cnt}/{self.record_cnt}({ratio:.2f}%)"
            )

    def update_record(self) -> None:
        self._try_write_errors()

        if self._has_record_content():
            if not self.error_flag:
                self._set_record()
                self.success_cnt += 1
            self.record_cnt += 1

        self._cleanup_column_names()
        self._update_progress_postfix()
        self._reset_record()

    def get_record_data(
        self,
    ) -> tuple[dict[str, list[Any]], list[str], dict[str, list[Any]], list[int], list[str]]:
        self.update_record()
        return (
            self.all_cols,
            self.all_col_names,
            self.all_peak,
            self.offsets,
            self.all_peak_meta_names,
        )

    def get_dataset(self) -> MSDataset:
        all_cols, all_col_names, all_peak, offsets, all_peak_meta_names = self.get_record_data()

        spectrum_metadata = pd.DataFrame(all_cols, columns=all_col_names)

        peak_data = np.column_stack(
            [
                np.asarray(all_peak["mz"], dtype=np.float64),
                np.asarray(all_peak["intensity"], dtype=np.float64),
            ]
        )
        offsets_array = np.asarray(offsets, dtype=np.int64)

        peak_metadata = None
        if all_peak_meta_names:
            peak_metadata = pd.DataFrame(
                {key: all_peak[key] for key in all_peak_meta_names},
                columns=all_peak_meta_names,
            )

        peak_series = PeakSeries(
            data=peak_data,
            offsets=offsets_array,
            metadata=peak_metadata,
            metadata_columns=all_peak_meta_names if all_peak_meta_names else None,
        )

        return MSDataset(
            spectrum_metadata=spectrum_metadata,
            peak_series=peak_series,
        )

    def add_meta(self, key: str, value: Any) -> tuple[str, Any]:
        parsed_key, parsed_value = self.item_parser.parse_item_pair(key, str(value))

        if parsed_key in self.meta:
            if not self.allow_duplicate_cols:
                raise ValueError(
                    f"Duplicate meta key: ({key} & {self.meta[parsed_key]}) -> {parsed_key}"
                )

            idx = 1
            while f"{parsed_key}{idx}" in self.meta:
                idx += 1
            parsed_key = f"{parsed_key}{idx}"

        if parsed_key not in self.all_cols:
            self.header_map[parsed_key] = key
            self.all_col_names.append(parsed_key)

        self.meta[parsed_key] = parsed_value
        return parsed_key, parsed_value

    def add_peak(self, mz: float, intensity: float, **metadata: Any) -> dict[str, Any]:
        peak_entry: dict[str, Any] = {
            "mz": mz,
            "intensity": intensity,
            **metadata,
        }

        for key in peak_entry:
            if key not in self.peak:
                self.peak[key] = [""] * len(self.peak["mz"])

        for key in self.peak:
            self.peak[key].append(peak_entry.get(key, ""))

        return peak_entry

    def add_error_message(self, message: str, line_text: str) -> None:
        self.error_flag = True
        message = message.strip().replace("\n", "\\n")
        line_text = line_text.strip().replace("\n", "\\n")
        self.error_text_list.append(
            f"[ERROR] Line ({self.line_count:05d}) {line_text} | {message}"
        )

    def _try_write_errors(self) -> bool:
        if self.error_flag and self.error_log_level != ErrorLogLevel.NONE:
            if self.error_log_level == ErrorLogLevel.DETAIL and self.recent_lines:
                self.error_text_list.append("[CONTEXT]")
                self.error_text_list.extend(self.recent_lines)

            error_text = "\n".join(self.error_text_list)

            if not os.path.exists(self.error_file_path):
                with open(self.error_file_path, "w", encoding=self.encoding):
                    pass

            with open(self.error_file_path, "a", encoding=self.encoding) as ef:
                ef.write(error_text + "\n\n")
            return True

        return False


def is_peak_header_line(line: str) -> bool:
    """
    Return whether a line is a peak header line.

    A valid peak header line must begin with at least:
    ``mz intensity``.
    """
    items = line.strip().split()
    return len(items) >= 2 and items[0].lower() == "mz" and items[1].lower() == "intensity"


def split_peak_metadata_text(metadata_text: str, line: str) -> list[str]:
    """
    Split peak metadata text by semicolon while preserving quoted substrings.
    """
    semi_split_fields = metadata_text.split(";")

    quote_start_idx = -1
    quote_end_idx = -1
    quote_char = ""
    merged_item = ""
    meta_items: list[str] = []

    for i, item in enumerate(semi_split_fields):
        stripped = item.strip()

        if stripped.startswith('"') and quote_start_idx == -1:
            quote_start_idx = i
            quote_char = '"'
        elif stripped.startswith("'") and quote_start_idx == -1:
            quote_start_idx = i
            quote_char = "'"

        if quote_start_idx != -1 and stripped.endswith(quote_char):
            quote_end_idx = i

        if quote_start_idx != -1 and quote_end_idx != -1:
            merged_item = ";".join(semi_split_fields[quote_start_idx:quote_end_idx + 1])
            merged_item = merged_item.strip().strip(quote_char)
            quote_start_idx = -1
            quote_end_idx = -1
            quote_char = ""

        elif quote_start_idx != -1 and quote_end_idx == -1:
            continue

        else:
            merged_item = item

        if quote_start_idx == -1:
            meta_items.append(merged_item.strip())
            merged_item = ""

    if merged_item != "":
        raise ValueError(
            f"Peak line '{line.strip()}' has unmatched quotes in metadata."
        )

    return meta_items


def parse_peak_line(
    line: str,
    *,
    peak_columns: Optional[List[str]] = None,
    auto_col_prefix: str = "column",
) -> Dict[str, Any]:
    """
    Parse one peak line into a peak entry dictionary.

    Parameters
    ----------
    line
        One peak row.
    peak_columns
        Column names for peak fields. If omitted, only ``mz`` and ``intensity``
        are assumed.
    auto_col_prefix
        Prefix for automatically generated metadata column names.

    Returns
    -------
    dict
        Parsed peak entry.

    Examples
    --------
    >>> parse_peak_line("100.0 200.0")
    {'mz': 100.0, 'intensity': 200.0}

    >>> parse_peak_line("100.0 200.0 fragA ; note1", peak_columns=["mz", "intensity", "frag", "note"])
    {'mz': 100.0, 'intensity': 200.0, 'frag': 'fragA', 'note': 'note1'}
    """
    if peak_columns is None:
        peak_columns = ["mz", "intensity"]

    items = line.strip().split(maxsplit=2)

    if len(items) == 2:
        mz_item, intensity_item = items
        meta_items: list[str] = []

    elif len(items) == 3:
        mz_item, intensity_item, metadata_text = items
        meta_items = split_peak_metadata_text(metadata_text, line)

    else:
        raise ValueError(
            f"Peak line '{line.strip()}' does not have valid m/z and intensity values."
        )

    peak_entry: Dict[str, Any] = {
        "mz": float(mz_item),
        "intensity": float(intensity_item),
    }

    for i, value in enumerate(meta_items):
        if i + 2 >= len(peak_columns):
            col = f"{auto_col_prefix}{i + 3 - len(peak_columns)}"
        else:
            col = peak_columns[i + 2]

        if value != "":
            if col in peak_entry:
                raise ValueError(
                    f"Duplicate peak metadata column '{col}' in line '{line.strip()}'."
                )
            peak_entry[col] = value

    return peak_entry

def _stringify_msp_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _quote_peak_meta_item(value: Any) -> str:
    text = _stringify_msp_value(value)
    if text == "":
        return ""
    return text.replace('"', "'")