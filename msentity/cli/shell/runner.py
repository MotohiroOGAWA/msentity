from __future__ import annotations

import shlex

from msentity import MSDataset, load_ms_dataset
from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.history import setup_history
from msentity.cli.shell.registry import build_command_map, discover_commands
from msentity.cli.shell.state import ShellState


class DatasetShell:
    """Interactive shell for exploring and processing MSDataset."""

    def __init__(
        self,
        dataset: MSDataset,
        input_file: str,
    ) -> None:
        self.state = ShellState(
            dataset=dataset,
            input_file=input_file,
        )

        self.commands = discover_commands()
        self.command_map = build_command_map(self.commands)

    def run(self) -> None:
        setup_history(
            completions=self.command_map.keys(),
        )

        print("msentity shell")
        print("Type 'help' to show commands. Type 'exit' to quit.")
        print()

        while True:
            try:
                line = input("msentity> ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue

            if not line:
                continue

            try:
                should_continue = self.run_line(line)
            except Exception as exc:
                print(f"Error: {exc}")
                continue

            if not should_continue:
                break

    def run_line(
        self,
        line: str,
    ) -> bool:
        tokens = shlex.split(line)

        if not tokens:
            return True

        command_name = tokens[0]
        args = tokens[1:]

        if command_name in {"help", "?"}:
            return self.run_help(args)

        command = self.command_map.get(command_name)

        if command is None:
            print(f"Unknown command: {command_name}")
            print("Type 'help' to show commands.")
            return True

        if wants_help(args):
            print(command.format_full_help())
            return True

        return command.run(self.state, args)

    def run_help(
        self,
        args: list[str],
    ) -> bool:
        if not args:
            print(self.format_general_help())
            return True

        command_name = args[0]
        command = self.command_map.get(command_name)

        if command is None:
            print(f"Unknown command: {command_name}")
            return True

        print(command.format_full_help())
        return True

    def format_general_help(self) -> str:
        lines: list[str] = []

        lines.append("Available commands:")
        lines.append("")

        for command in self.commands:
            lines.append(command.format_short_help())
            lines.append("")

        lines.append("Use '<command> --help' or 'help <command>' for details.")

        return "\n".join(lines).rstrip()


def wants_help(args: list[str]) -> bool:
    return "--help" in args or "-h" in args


def run_shell(
    input_file: str,
    *,
    file_type: str | None = None,
    spec_id_prefix: str | None = None,
) -> None:
    dataset = load_ms_dataset(
        input_file,
        file_type=file_type,
        spec_id_prefix=spec_id_prefix,
    )

    shell = DatasetShell(
        dataset=dataset,
        input_file=input_file,
    )
    shell.run()