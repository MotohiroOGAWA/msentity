from __future__ import annotations

import pandas as pd

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class PeaksCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="peaks",
            usage="peaks <index> [--top n] [--sort mz|intensity] [--ascending]",
            summary="Show peaks for one spectrum.",
            description=(
                "Show the peak table of one spectrum. "
                "The output contains m/z and intensity columns."
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
                    "--sort mz|intensity",
                    "Sort peaks by m/z or intensity.",
                ),
                (
                    "--ascending",
                    "Sort in ascending order. By default, sorting is descending.",
                ),
            ],
            examples=[
                "peaks 0",
                "peaks 0 --top 10",
                "peaks 0 --top 10 --sort intensity",
                "peaks 0 --sort mz --ascending",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
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


def validate_spectrum_index(
    state: ShellState,
    index: int,
) -> None:
    if not (0 <= index < len(state.dataset)):
        raise IndexError(f"spectrum index out of range: {index}")


def get_command() -> ShellCommand:
    return PeaksCommand()