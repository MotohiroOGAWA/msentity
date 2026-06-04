from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class ResetCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="reset",
            usage="reset",
            summary="Reset to the full dataset view.",
            description=(
                "Reset the current dataset view to all spectra. "
                "This uses MSDataset.reset_view()."
            ),
            examples=[
                "reset",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        state.dataset.reset_view()

        print(f"Reset dataset view: {len(state.dataset)} spectra")

        return True


def get_command() -> ShellCommand:
    return ResetCommand()