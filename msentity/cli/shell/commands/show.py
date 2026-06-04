from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class ShowCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="show",
            usage="show <index>",
            summary="Show metadata for one spectrum.",
            description=(
                "Show spectrum-level metadata for one spectrum."
            ),
            arguments=[
                (
                    "index",
                    "Spectrum index in the current dataset view.",
                ),
            ],
            examples=[
                "show 0",
                "show 12",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if len(args) != 1:
            raise ValueError("Usage: show <index>")

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
    return ShowCommand()