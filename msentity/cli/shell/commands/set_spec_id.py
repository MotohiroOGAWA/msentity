from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState
from msentity.processing.id import set_spec_id


class SetSpecIDCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="specid",
            usage="specid [--col-name COL_NAME] [--prefix PREFIX] [--start START] [--overwrite]",
            summary="Assign sequential spectrum IDs to the dataset.",
            description=(
                "Create a spectrum-level metadata column containing sequential "
                "spectrum identifiers. By default, this creates the 'SpecID' column."
            ),
            options=[
                (
                    "--col-name COL_NAME",
                    "Name of the spectrum metadata column to create. Default: SpecID.",
                ),
                (
                    "--prefix PREFIX",
                    "Prefix added before each numeric spectrum ID. Default: empty string.",
                ),
                (
                    "--start START",
                    "Starting number for spectrum IDs. Default: 1.",
                ),
                (
                    "--overwrite",
                    "Overwrite the column if it already exists.",
                ),
            ],
            examples=[
                "specid",
                "specid --prefix SP",
                "specid --col-name SpectrumID --prefix S --start 1",
                "specid --overwrite",
            ],
            aliases=("set_spec_id", "spec_id"),
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        options = _parse_set_spec_id_args(args)

        changed = set_spec_id(
            state.dataset,
            col_name=options.col_name,
            prefix=options.prefix,
            overwrite=options.overwrite,
            start=options.start,
        )

        if changed:
            print(
                f"Assigned spectrum IDs: column='{options.col_name}', "
                f"prefix='{options.prefix}', start={options.start}"
            )
        else:
            print(
                f"Skipped spectrum ID assignment: column '{options.col_name}' already exists. "
                "Use --overwrite to replace it."
            )

        return True


class _SetSpecIDOptions:
    def __init__(
        self,
        *,
        col_name: str = "SpecID",
        prefix: str = "",
        start: int = 1,
        overwrite: bool = False,
    ) -> None:
        self.col_name = col_name
        self.prefix = prefix
        self.start = start
        self.overwrite = overwrite


def _parse_set_spec_id_args(args: list[str]) -> _SetSpecIDOptions:
    col_name = "SpecID"
    prefix = ""
    start = 1
    overwrite = False

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "--col-name":
            i += 1
            if i >= len(args):
                raise ValueError("Missing value for --col-name")
            col_name = args[i]

        elif arg == "--prefix":
            i += 1
            if i >= len(args):
                raise ValueError("Missing value for --prefix")
            prefix = args[i]

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
            raise ValueError(f"Unknown option for specid: {arg}")

        i += 1

    return _SetSpecIDOptions(
        col_name=col_name,
        prefix=prefix,
        start=start,
        overwrite=overwrite,
    )


def get_command() -> ShellCommand:
    return SetSpecIDCommand()