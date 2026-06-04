from __future__ import annotations

from pathlib import Path

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class ExportCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="export",
            usage="export <output.msds>",
            summary="Export the current dataset view.",
            description=(
                "Export the current dataset view as an .msds file."
            ),
            arguments=[
                (
                    "output.msds",
                    "Output file path.",
                ),
            ],
            examples=[
                "export filtered.msds",
                "export results/filtered.msds",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if len(args) != 1:
            raise ValueError("Usage: export <output.msds>")

        output_file = Path(args[0])

        if output_file.suffix.lower() != ".msds":
            raise ValueError("export currently supports .msds output only")

        state.dataset.save(str(output_file))

        print(f"Exported: {output_file}")

        return True


def get_command() -> ShellCommand:
    return ExportCommand()