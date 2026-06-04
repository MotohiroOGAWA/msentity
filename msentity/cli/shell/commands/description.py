from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class DescriptionCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="description",
            usage="description [value...]",
            summary="Show or set dataset description.",
            description=(
                "Show the current dataset description when no value is given. "
                "Set the dataset description when a value is given."
            ),
            arguments=[
                (
                    "value",
                    "New dataset description. If omitted, the current description is shown.",
                ),
            ],
            examples=[
                "description",
                "description Palm oil LC-MS/MS dataset",
                'description "Palm oil LC-MS/MS dataset"',
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if not args:
            print(state.dataset.description)
            return True

        value = " ".join(args)
        state.dataset.description = value

        print(f"description: {state.dataset.description}")

        return True


def get_command() -> ShellCommand:
    return DescriptionCommand()