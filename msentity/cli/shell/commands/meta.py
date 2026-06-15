from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class MetaCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="meta",
            usage="meta <index>",
            summary="Show metadata for one spectrum.",
            description=(
                "Show spectrum-level metadata for one spectrum. "
                "This command does not show peak data."
            ),
            arguments=[
                (
                    "index",
                    "Spectrum index in the current dataset view.",
                ),
            ],
            examples=[
                "meta 0",
                "meta 12",
            ],
            aliases=("metadata",),
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if len(args) != 1:
            raise ValueError("Usage: meta <index>")

        index = int(args[0])
        validate_spectrum_index(state, index)

        row = state.dataset.metadata.iloc[index]
        print(row.to_string())

        return True


def validate_spectrum_index(
    state: ShellState,
    index: int,
) -> None:
    if not (0 <= index < len(state.dataset)):
        raise IndexError(f"spectrum index out of range: {index}")


def get_command() -> ShellCommand:
    return MetaCommand()