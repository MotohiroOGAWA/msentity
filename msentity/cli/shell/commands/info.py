from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class InfoCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="info",
            usage="info",
            summary="Show dataset summary.",
            description=(
                "Show basic information about the current dataset view."
            ),
            examples=[
                "info",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        dataset = state.dataset

        print(f"input_file: {state.input_file}")
        print(f"n_spectra: {len(dataset)}")
        print(f"n_columns: {dataset.n_columns}")
        print(f"n_peaks_total: {dataset.n_peaks_total}")
        print(f"description: {dataset.description}")
        print(f"attributes: {dataset.attributes}")
        print(f"tags: {dataset.tags}")

        return True


def get_command() -> ShellCommand:
    return InfoCommand()