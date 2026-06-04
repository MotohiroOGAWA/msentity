from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class ExitCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="exit",
            usage="exit",
            summary="Quit the shell.",
            description=(
                "Quit the msentity shell."
            ),
            aliases=("quit",),
            examples=[
                "exit",
                "quit",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        return False


def get_command() -> ShellCommand:
    return ExitCommand()