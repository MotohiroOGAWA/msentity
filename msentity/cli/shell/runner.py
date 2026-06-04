from __future__ import annotations

import shlex
from typing import Callable

from msentity import MSDataset, load_ms_dataset
from msentity.cli.shell.commands import (
    command_columns,
    command_exit,
    command_filter,
    command_head,
    command_help,
    command_info,
    command_normalize,
    command_peaks,
    command_reset,
    command_save,
    command_show,
    command_sort,
)
from msentity.cli.shell.history import setup_history
from msentity.cli.shell.state import ShellState


ShellCommand = Callable[[ShellState, list[str]], bool]


class DatasetShell:
    """Interactive shell for exploring and processing MSDataset."""

    def __init__(
        self,
        dataset: MSDataset,
        input_file: str,
    ) -> None:
        self.state = ShellState(
            original_dataset=dataset,
            dataset=dataset,
            input_file=input_file,
        )
        self.command_map = build_command_map()

    def run(self) -> None:
        setup_history()
        
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

    def run_line(self, line: str) -> bool:
        tokens = shlex.split(line)

        if not tokens:
            return True

        command_name = tokens[0]
        args = tokens[1:]

        command = self.command_map.get(command_name)

        if command is None:
            print(f"Unknown command: {command_name}")
            print("Type 'help' to show commands.")
            return True

        return command(self.state, args)


def build_command_map() -> dict[str, ShellCommand]:
    return {
        "help": command_help,
        "?": command_help,
        "info": command_info,
        "columns": command_columns,
        "head": command_head,
        "show": command_show,
        "peaks": command_peaks,
        "filter": command_filter,
        "sort": command_sort,
        "normalize": command_normalize,
        "save": command_save,
        "reset": command_reset,
        "exit": command_exit,
        "quit": command_exit,
    }


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