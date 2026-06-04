from __future__ import annotations

from pathlib import Path

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class ExportCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="export",
            usage="export [output.msds]",
            summary="Export the current dataset view.",
            description=(
                "Export the current dataset view as an .msds file. "
                "If output.msds is omitted, the output path is generated "
                "from the input file path by replacing its suffix with .msds."
            ),
            arguments=[
                (
                    "output.msds",
                    "Output file path. If omitted, the input file name with .msds is used.",
                ),
            ],
            examples=[
                "export",
                "export filtered.msds",
                "export results/filtered.msds",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if len(args) > 1:
            raise ValueError("Usage: export [output.msds]")

        output_file = self._resolve_output_file(
            state=state,
            args=args,
        )

        if output_file.suffix.lower() != ".msds":
            raise ValueError("export currently supports .msds output only")

        print(f"Output file: {output_file}")

        if not confirm("Export to this file?"):
            print("Canceled export.")
            return True

        if output_file.exists():
            if not confirm("File already exists. Overwrite it?"):
                print("Canceled export.")
                return True

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        state.dataset.save(str(output_file))

        print(f"Exported: {output_file}")

        return True

    def _resolve_output_file(
        self,
        *,
        state: ShellState,
        args: list[str],
    ) -> Path:
        if args:
            return Path(args[0])

        input_file = Path(state.input_file)

        if input_file.suffix:
            return input_file.with_suffix(".msds")

        return input_file.with_name(f"{input_file.name}.msds")


def confirm(
    message: str,
    *,
    default: bool = False,
) -> bool:
    suffix = "[y/N]" if not default else "[Y/n]"

    while True:
        answer = input(f"{message} {suffix} ").strip().lower()

        if not answer:
            return default

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("Please enter y or n.")


def get_command() -> ShellCommand:
    return ExportCommand()