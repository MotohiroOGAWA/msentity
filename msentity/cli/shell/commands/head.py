from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class HeadCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="head",
            usage="head [n]",
            summary="Show first n metadata rows.",
            description=(
                "Show the first n rows of spectrum-level metadata."
            ),
            arguments=[
                (
                    "n",
                    "Number of rows to show. Default: 5.",
                ),
            ],
            examples=[
                "head",
                "head 10",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        n_rows = 5

        if len(args) >= 1:
            n_rows = int(args[0])

        metadata = state.dataset.metadata.head(n_rows)
        print(metadata.to_string(index=True))

        return True


def get_command() -> ShellCommand:
    return HeadCommand()