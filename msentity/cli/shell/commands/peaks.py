from __future__ import annotations

import pandas as pd

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class PeaksCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="peaks",
            usage=(
                "peaks <index> [--top n] [--sort mz|intensity|metadata_column] "
                "[--ascending] [--no-metadata]"
            ),
            summary="Show peaks for one spectrum.",
            description=(
                "Show the peak table of one spectrum. "
                "The output contains m/z and intensity columns. "
                "If peak-level metadata is available, metadata columns are also displayed."
            ),
            arguments=[
                (
                    "index",
                    "Spectrum index in the current dataset view.",
                ),
            ],
            options=[
                (
                    "--top n",
                    "Show only the first n peaks after optional sorting.",
                ),
                (
                    "--sort mz|intensity|metadata_column",
                    "Sort peaks by m/z, intensity, or a peak metadata column.",
                ),
                (
                    "--ascending",
                    "Sort in ascending order. By default, sorting is descending.",
                ),
                (
                    "--no-metadata",
                    "Hide peak-level metadata columns even if they are available.",
                ),
            ],
            examples=[
                "peaks 0",
                "peaks 0 --top 10",
                "peaks 0 --top 10 --sort intensity",
                "peaks 0 --sort mz --ascending",
                "peaks 0 --sort PeakID --ascending",
                "peaks 0 --no-metadata",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if not args:
            raise ValueError(
                "Usage: peaks <index> [--top n] [--sort mz|intensity|metadata_column] "
                "[--ascending] [--no-metadata]"
            )

        index = int(args[0])
        validate_spectrum_index(state, index)

        top_n: int | None = None
        sort_column: str | None = None
        ascending = False
        show_metadata = True

        i = 1
        while i < len(args):
            token = args[i]

            if token == "--top":
                i += 1
                if i >= len(args):
                    raise ValueError("--top requires an integer")
                top_n = int(args[i])
                if top_n < 0:
                    raise ValueError("--top must be greater than or equal to 0")

            elif token == "--sort":
                i += 1
                if i >= len(args):
                    raise ValueError("--sort requires a column name")
                sort_column = args[i]

            elif token == "--ascending":
                ascending = True

            elif token == "--no-metadata":
                show_metadata = False

            else:
                raise ValueError(f"Unknown option for peaks: {token}")

            i += 1

        spectrum = state.dataset.peaks[index]
        peaks = build_peak_table(
            spectrum,
            show_metadata=show_metadata,
        )

        if sort_column is not None:
            if sort_column not in peaks.columns:
                raise ValueError(
                    f"--sort column not found: {sort_column}. "
                    f"Available columns: {', '.join(map(str, peaks.columns))}"
                )

            peaks = peaks.sort_values(
                sort_column,
                ascending=ascending,
                kind="stable",
            )

        if top_n is not None:
            peaks = peaks.head(top_n)

        print(peaks.to_string(index=False))

        return True


def build_peak_table(
    spectrum,
    *,
    show_metadata: bool = True,
) -> pd.DataFrame:
    """
    Build a display table for one spectrum.

    The output always contains ``mz`` and ``intensity``.
    If peak-level metadata exists, metadata columns are appended.
    """
    peaks = pd.DataFrame(
        spectrum.data,
        columns=["mz", "intensity"],
    )

    if not show_metadata:
        return peaks

    metadata = getattr(spectrum, "metadata", None)

    if metadata is None:
        return peaks

    if len(metadata) == 0:
        return peaks

    metadata = metadata.reset_index(drop=True)

    if len(metadata) != len(peaks):
        raise ValueError(
            "Peak metadata row count does not match peak data row count: "
            f"{len(metadata)} != {len(peaks)}"
        )

    # Avoid duplicate columns if metadata accidentally contains mz/intensity.
    metadata = metadata.drop(
        columns=[col for col in ["mz", "intensity"] if col in metadata.columns],
        errors="ignore",
    )

    if metadata.shape[1] == 0:
        return peaks

    return pd.concat(
        [
            peaks.reset_index(drop=True),
            metadata,
        ],
        axis=1,
    )


def validate_spectrum_index(
    state: ShellState,
    index: int,
) -> None:
    if not (0 <= index < len(state.dataset)):
        raise IndexError(f"spectrum index out of range: {index}")


def get_command() -> ShellCommand:
    return PeaksCommand()