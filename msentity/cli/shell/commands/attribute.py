from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class AttributeCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="attribute",
            usage="attribute <list|get|set|remove|clear> [key] [value...]",
            summary="Show or edit dataset attributes.",
            description=(
                "Manage dataset-level attributes. "
                "Attributes are string key-value pairs stored in the MSDataset."
            ),
            arguments=[
                (
                    "subcommand",
                    "One of: list, get, set, remove, clear.",
                ),
                (
                    "key",
                    "Attribute key.",
                ),
                (
                    "value",
                    "Attribute value for the set subcommand.",
                ),
            ],
            examples=[
                "attribute list",
                "attribute set source unit-test",
                'attribute set source "MassBank export"',
                "attribute get source",
                "attribute remove source",
                "attribute clear",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if not args:
            raise ValueError("Usage: attribute <list|get|set|remove|clear> [key] [value...]")

        subcommand = args[0]

        if subcommand == "list":
            return self._list_attributes(state)

        if subcommand == "get":
            return self._get_attribute(state, args[1:])

        if subcommand == "set":
            return self._set_attribute(state, args[1:])

        if subcommand == "remove":
            return self._remove_attribute(state, args[1:])

        if subcommand == "clear":
            return self._clear_attributes(state, args[1:])

        raise ValueError("Unknown attribute subcommand. Use one of: list, get, set, remove, clear")

    def _list_attributes(
        self,
        state: ShellState,
    ) -> bool:
        attributes = state.dataset.attributes

        if not attributes:
            print("{}")
            return True

        for key, value in attributes.items():
            print(f"{key}: {value}")

        return True

    def _get_attribute(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if len(args) != 1:
            raise ValueError("Usage: attribute get <key>")

        key = args[0]
        attributes = state.dataset.attributes

        if key not in attributes:
            raise KeyError(f"attribute not found: {key}")

        print(attributes[key])

        return True

    def _set_attribute(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if len(args) < 2:
            raise ValueError("Usage: attribute set <key> <value...>")

        key = args[0]
        value = " ".join(args[1:])

        state.dataset.set_attribute(key, value)

        print(f"{key}: {value}")

        return True

    def _remove_attribute(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if len(args) != 1:
            raise ValueError("Usage: attribute remove <key>")

        key = args[0]
        removed = state.dataset.remove_attribute(key)

        if removed:
            print(f"Removed attribute: {key}")
        else:
            print(f"Attribute not found: {key}")

        return True

    def _clear_attributes(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if args:
            raise ValueError("Usage: attribute clear")

        state.dataset.clear_attributes()
        print("Cleared attributes")

        return True


def get_command() -> ShellCommand:
    return AttributeCommand()