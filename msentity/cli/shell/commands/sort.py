from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class SortCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="sort",
            usage="sort <column> [asc|desc]",
            summary="Sort spectra by metadata column.",
            description=(
                "Sort the current dataset view by a spectrum metadata column."
            ),
            arguments=[
                (
                    "column",
                    "Metadata column name.",
                ),
                (
                    "asc|desc",
                    "Sort order. Default: asc.",
                ),
            ],
            examples=[
                "sort PrecursorMZ",
                "sort PrecursorMZ desc",
                "sort Name asc",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if not args:
            raise ValueError("Usage: sort <column> [asc|desc]")

        column = args[0]
        order = args[1] if len(args) >= 2 else "asc"

        if order not in {"asc", "desc"}:
            raise ValueError("sort order must be 'asc' or 'desc'")

        state.dataset = state.dataset.sort_by(
            column,
            ascending=(order == "asc"),
        )

        print(f"Sorted by {column} ({order})")

        return True


def get_command() -> ShellCommand:
    return SortCommand()