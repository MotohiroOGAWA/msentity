from __future__ import annotations

import importlib
import pkgutil

import msentity.cli.shell.commands as commands_package
from msentity.cli.shell.command import ShellCommand


def discover_commands() -> list[ShellCommand]:
    """Discover shell commands from msentity.cli.shell.commands."""
    commands: list[ShellCommand] = []

    for module_info in pkgutil.iter_modules(
        commands_package.__path__,
        prefix=f"{commands_package.__name__}.",
    ):
        module = importlib.import_module(module_info.name)

        get_command = getattr(module, "get_command", None)
        if get_command is None:
            continue

        command = get_command()

        if not isinstance(command, ShellCommand):
            raise TypeError(
                f"{module_info.name}.get_command() must return ShellCommand"
            )

        commands.append(command)

    commands.sort(key=lambda command: command.name)

    return commands


def build_command_map(
    commands: list[ShellCommand],
) -> dict[str, ShellCommand]:
    command_map: dict[str, ShellCommand] = {}

    for command in commands:
        if command.name in command_map:
            raise ValueError(f"Duplicate shell command: {command.name}")

        command_map[command.name] = command

        for alias in command.aliases:
            if alias in command_map:
                raise ValueError(f"Duplicate shell command alias: {alias}")

            command_map[alias] = command

    return command_map