from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def add_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Discover and add CLI commands.

    Command files live in msentity.cli.commands. Command packages live next to
    msentity.cli.commands, such as msentity.cli.shell.
    """
    commands_path = Path(__file__).parent
    cli_path = commands_path.parent

    _add_command_files(__name__, commands_path, subparsers)
    _add_command_packages("msentity.cli", cli_path, subparsers)


def _add_command_files(
    package_name: str,
    package_path: Path,
    subparsers: argparse._SubParsersAction,
) -> None:
    for path in sorted(package_path.iterdir(), key=lambda item: item.name):
        if path.name.startswith("_") or path.suffix != ".py":
            continue

        _add_command_module(f"{package_name}.{path.stem}", subparsers)


def _add_command_packages(
    package_name: str,
    package_path: Path,
    subparsers: argparse._SubParsersAction,
) -> None:
    for path in sorted(package_path.iterdir(), key=lambda item: item.name):
        if path.name.startswith("_") or path.name == "commands":
            continue

        if not path.is_dir() or not (path / "__init__.py").is_file():
            continue

        _add_command_module(f"{package_name}.{path.name}", subparsers)


def _add_command_module(
    module_name: str,
    subparsers: argparse._SubParsersAction,
) -> None:
    module = importlib.import_module(module_name)
    setup_parser = getattr(module, "setup_parser", None)

    if setup_parser is None:
        return

    setup_parser(subparsers)


from msentity.cli.commands.convert import setup_parser as add_convert_command
from msentity.cli.commands.head import setup_parser as add_head_command
from msentity.cli.commands.info import setup_parser as add_info_command
from msentity.cli.commands.merge_dir import setup_parser as add_merge_dir_command
from msentity.cli.commands.meta import setup_parser as add_meta_command
from msentity.cli.shell import setup_parser as add_shell_command

__all__ = [
    "add_commands",
    "add_convert_command",
    "add_head_command",
    "add_info_command",
    "add_merge_dir_command",
    "add_meta_command",
    "add_shell_command",
]
