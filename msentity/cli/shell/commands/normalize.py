from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class NormalizeCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="normalize",
            usage="normalize [scale]",
            summary="Normalize peak intensities.",
            description=(
                "Normalize peak intensities so that the maximum intensity "
                "of each spectrum becomes the given scale."
            ),
            arguments=[
                (
                    "scale",
                    "Target maximum intensity for each spectrum. Default: 1.0.",
                ),
            ],
            examples=[
                "normalize",
                "normalize 100",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        scale = 1.0

        if args:
            scale = float(args[0])

        state.dataset.peaks.normalize(
            scale=scale,
            in_place=True,
        )

        print(f"Normalized intensities with scale={scale}")

        return True


def get_command() -> ShellCommand:
    return NormalizeCommand()