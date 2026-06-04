from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from msentity.cli.shell.state import ShellState


@dataclass
class ShellCommand(ABC):
    """Base class for one shell command."""

    name: str
    usage: str
    summary: str
    description: str = ""
    arguments: list[tuple[str, str]] = field(default_factory=list)
    options: list[tuple[str, str]] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    aliases: tuple[str, ...] = ()

    def matches(self, name: str) -> bool:
        return name == self.name or name in self.aliases

    def format_short_help(self) -> str:
        return f"  {self.usage}\n      {self.summary}"

    def format_full_help(self) -> str:
        lines: list[str] = []

        lines.append("Usage:")
        lines.append(f"  {self.usage}")

        if self.aliases:
            lines.append("")
            lines.append("Aliases:")
            lines.append(f"  {', '.join(self.aliases)}")

        if self.description:
            lines.append("")
            lines.append("Description:")
            lines.extend(_indent_lines(self.description))

        if self.arguments:
            lines.append("")
            lines.append("Arguments:")
            for name, text in self.arguments:
                lines.append(f"  {name}")
                lines.extend(_indent_lines(text, indent="      "))

        if self.options:
            lines.append("")
            lines.append("Options:")
            for name, text in self.options:
                lines.append(f"  {name}")
                lines.extend(_indent_lines(text, indent="      "))

        if self.examples:
            lines.append("")
            lines.append("Examples:")
            for example in self.examples:
                lines.append(f"  {example}")

        return "\n".join(lines)

    @abstractmethod
    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        """
        Execute the command.

        Returns
        -------
        bool
            True to continue the shell, False to exit.
        """


def _indent_lines(
    text: str,
    *,
    indent: str = "  ",
) -> list[str]:
    return [
        f"{indent}{line}" if line else ""
        for line in text.strip().splitlines()
    ]