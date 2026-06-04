from __future__ import annotations

from pathlib import Path

import pandas as pd

from msentity.cli.shell.filters import build_filter_mask
from msentity.cli.shell.state import ShellState


def command_help(state: ShellState, args: list[str]) -> bool:
    print(
        """
Available commands:

  help
      Show this help message.

  info
      Show dataset summary.

  columns
      Show metadata columns.

  head [n]
      Show first n metadata rows.
      Example:
        head
        head 10

  show <index>
      Show metadata for one spectrum.
      Example:
        show 0

  peaks <index> [--top n] [--sort mz|intensity] [--ascending]
      Show peaks for one spectrum.
      Example:
        peaks 0
        peaks 0 --top 10
        peaks 0 --top 10 --sort intensity
        peaks 0 --sort mz --ascending

  filter <column> <op> <value>
      Filter spectra by metadata.
      Supported ops:
        ==, !=, >, >=, <, <=, contains

      Examples:
        filter PrecursorMZ > 300
        filter Name contains glucose
        filter IonMode == POSITIVE

  sort <column> [asc|desc]
      Sort spectra by metadata column.
      Example:
        sort PrecursorMZ desc

  normalize [scale]
      Normalize peak intensities in the current dataset.
      Example:
        normalize
        normalize 100

  save <output.msds>
      Save the current dataset view.
      Example:
        save filtered.msds

  reset
      Reset to the original loaded dataset.

  exit
      Quit the shell.
""".strip()
    )

    return True


def command_info(state: ShellState, args: list[str]) -> bool:
    dataset = state.dataset

    print(f"input_file: {state.input_file}")
    print(f"n_spectra: {len(dataset)}")
    print(f"n_columns: {dataset.n_columns}")
    print(f"n_peaks_total: {dataset.n_peaks_total}")
    print(f"description: {dataset.description}")
    print(f"attributes: {dataset.attributes}")
    print(f"tags: {dataset.tags}")

    return True


def command_columns(state: ShellState, args: list[str]) -> bool:
    for column in state.dataset.columns:
        print(column)

    return True


def command_head(state: ShellState, args: list[str]) -> bool:
    n_rows = 5

    if len(args) >= 1:
        n_rows = int(args[0])

    metadata = state.dataset.metadata.head(n_rows)
    print(metadata.to_string(index=True))

    return True


def command_show(state: ShellState, args: list[str]) -> bool:
    if len(args) != 1:
        raise ValueError("Usage: show <index>")

    index = int(args[0])
    validate_spectrum_index(state, index)

    row = state.dataset.metadata.iloc[index]
    print(row.to_string())

    return True


def command_peaks(state: ShellState, args: list[str]) -> bool:
    if not args:
        raise ValueError(
            "Usage: peaks <index> [--top n] [--sort mz|intensity] [--ascending]"
        )

    index = int(args[0])
    validate_spectrum_index(state, index)

    top_n: int | None = None
    sort_column: str | None = None
    ascending = False

    i = 1
    while i < len(args):
        token = args[i]

        if token == "--top":
            i += 1
            if i >= len(args):
                raise ValueError("--top requires an integer")
            top_n = int(args[i])

        elif token == "--sort":
            i += 1
            if i >= len(args):
                raise ValueError("--sort requires 'mz' or 'intensity'")
            sort_column = args[i]

            if sort_column not in {"mz", "intensity"}:
                raise ValueError("--sort must be 'mz' or 'intensity'")

        elif token == "--ascending":
            ascending = True

        else:
            raise ValueError(f"Unknown option for peaks: {token}")

        i += 1

    spectrum = state.dataset.peaks[index]

    peaks = pd.DataFrame(
        spectrum.data,
        columns=["mz", "intensity"],
    )

    if sort_column is not None:
        peaks = peaks.sort_values(
            sort_column,
            ascending=ascending,
            kind="stable",
        )

    if top_n is not None:
        peaks = peaks.head(top_n)

    print(peaks.to_string(index=False))

    return True


def command_filter(state: ShellState, args: list[str]) -> bool:
    if len(args) < 3:
        raise ValueError("Usage: filter <column> <op> <value>")

    column = args[0]
    op = args[1]
    value = " ".join(args[2:])

    dataset = state.dataset

    if column not in dataset.columns:
        raise KeyError(f"column not found: {column}")

    series = dataset[column]
    mask = build_filter_mask(
        series=series,
        op=op,
        value=value,
    )

    state.dataset = dataset[mask.reset_index(drop=True)]

    print(f"Filtered dataset: {len(state.dataset)} spectra")

    return True


def command_sort(state: ShellState, args: list[str]) -> bool:
    if not args:
        raise ValueError("Usage: sort <column> [asc|desc]")

    column = args[0]
    order = args[1] if len(args) >= 2 else "asc"

    if order not in {"asc", "desc"}:
        raise ValueError("sort order must be 'asc' or 'desc'")

    ascending = order == "asc"

    state.dataset = state.dataset.sort_by(
        column,
        ascending=ascending,
    )

    print(f"Sorted by {column} ({order})")

    return True


def command_normalize(state: ShellState, args: list[str]) -> bool:
    scale = 1.0

    if args:
        scale = float(args[0])

    state.dataset.peaks.normalize(
        scale=scale,
        in_place=True,
    )

    print(f"Normalized intensities with scale={scale}")

    return True


def command_save(state: ShellState, args: list[str]) -> bool:
    if len(args) != 1:
        raise ValueError("Usage: save <output.msds>")

    output_file = Path(args[0])

    if output_file.suffix.lower() != ".msds":
        raise ValueError("save currently supports .msds output only")

    state.dataset.save(str(output_file))

    print(f"Saved: {output_file}")

    return True


def command_reset(state: ShellState, args: list[str]) -> bool:
    state.dataset = state.original_dataset

    print(f"Reset dataset: {len(state.dataset)} spectra")

    return True


def command_exit(state: ShellState, args: list[str]) -> bool:
    return False


def validate_spectrum_index(
    state: ShellState,
    index: int,
) -> None:
    if not (0 <= index < len(state.dataset)):
        raise IndexError(f"spectrum index out of range: {index}")