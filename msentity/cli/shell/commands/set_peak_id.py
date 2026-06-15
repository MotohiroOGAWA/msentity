from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState
from msentity.processing.id import set_peak_id


class SetPeakIDCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="peakid",
            usage="peakid [--col-name COL_NAME] [--start START] [--overwrite]",
            summary="Assign local peak IDs to each spectrum.",
            description=(
                "Create a peak-level metadata column containing local peak IDs. "
                "Peak IDs are assigned independently within each spectrum."
            ),
            options=[
                (
                    "--col-name COL_NAME",
                    "Name of the peak metadata column to create. Default: PeakID.",
                ),
                (
                    "--start START",
                    "Starting number for peak IDs within each spectrum. Default: 0.",
                ),
                (
                    "--overwrite",
                    "Overwrite the column if it already exists.",
                ),
            ],
            examples=[
                "peakid",
                "peakid --start 1",
                "peakid --col-name LocalPeakID --start 1",
                "peakid --overwrite",
            ],
            aliases=("set_peak_id", "peak_id"),
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        options = _parse_set_peak_id_args(args)

        changed = set_peak_id(
            state.dataset,
            col_name=options.col_name,
            overwrite=options.overwrite,
            start=options.start,
        )

        if changed:
            print(
                f"Assigned peak IDs: column='{options.col_name}', "
                f"start={options.start}"
            )
        else:
            print(
                f"Skipped peak ID assignment: column '{options.col_name}' already exists. "
                "Use --overwrite to replace it."
            )

        return True


class _SetPeakIDOptions:
    def __init__(
        self,
        *,
        col_name: str = "PeakID",
        start: int = 0,
        overwrite: bool = False,
    ) -> None:
        self.col_name = col_name
        self.start = start
        self.overwrite = overwrite


def _parse_set_peak_id_args(args: list[str]) -> _SetPeakIDOptions:
    col_name = "PeakID"
    start = 0
    overwrite = False

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "--col-name":
            i += 1
            if i >= len(args):
                raise ValueError("Missing value for --col-name")
            col_name = args[i]

        elif arg == "--start":
            i += 1
            if i >= len(args):
                raise ValueError("Missing value for --start")
            try:
                start = int(args[i])
            except ValueError as exc:
                raise ValueError("--start must be an integer") from exc

        elif arg == "--overwrite":
            overwrite = True

        else:
            raise ValueError(f"Unknown option for peakid: {arg}")

        i += 1

    return _SetPeakIDOptions(
        col_name=col_name,
        start=start,
        overwrite=overwrite,
    )


def get_command() -> ShellCommand:
    return SetPeakIDCommand()