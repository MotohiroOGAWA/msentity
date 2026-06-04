from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class ColumnsCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="columns",
            usage="columns",
            summary="Show metadata columns.",
            description=(
                "Show metadata column names in the current dataset view."
            ),
            examples=[
                "columns",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        for column in state.dataset.columns:
            print(column)

        return True


def get_command() -> ShellCommand:
    return ColumnsCommand()