from __future__ import annotations

from msentity.cli.shell.command import ShellCommand
from msentity.cli.shell.state import ShellState


class TagCommand(ShellCommand):
    def __init__(self) -> None:
        super().__init__(
            name="tag",
            usage="tag <list|add|remove|clear> [tag]",
            summary="Show or edit dataset tags.",
            description=(
                "Manage dataset-level tags. "
                "Tags are string labels stored in the MSDataset."
            ),
            arguments=[
                (
                    "subcommand",
                    "One of: list, add, remove, clear.",
                ),
                (
                    "tag",
                    "Tag value.",
                ),
            ],
            examples=[
                "tag list",
                "tag add test",
                "tag add processed",
                "tag remove test",
                "tag clear",
            ],
        )

    def run(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if not args:
            raise ValueError("Usage: tag <list|add|remove|clear> [tag]")

        subcommand = args[0]

        if subcommand == "list":
            return self._list_tags(state)

        if subcommand == "add":
            return self._add_tag(state, args[1:])

        if subcommand == "remove":
            return self._remove_tag(state, args[1:])

        if subcommand == "clear":
            return self._clear_tags(state, args[1:])

        raise ValueError("Unknown tag subcommand. Use one of: list, add, remove, clear")

    def _list_tags(
        self,
        state: ShellState,
    ) -> bool:
        tags = state.dataset.tags

        if not tags:
            print("[]")
            return True

        for tag in tags:
            print(tag)

        return True

    def _add_tag(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if len(args) != 1:
            raise ValueError("Usage: tag add <tag>")

        tag = args[0]
        added = state.dataset.add_tag(tag)

        if added:
            print(f"Added tag: {tag}")
        else:
            print(f"Tag already exists: {tag}")

        return True

    def _remove_tag(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if len(args) != 1:
            raise ValueError("Usage: tag remove <tag>")

        tag = args[0]
        removed = state.dataset.remove_tag(tag)

        if removed:
            print(f"Removed tag: {tag}")
        else:
            print(f"Tag not found: {tag}")

        return True

    def _clear_tags(
        self,
        state: ShellState,
        args: list[str],
    ) -> bool:
        if args:
            raise ValueError("Usage: tag clear")

        state.dataset.clear_tags()
        print("Cleared tags")

        return True


def get_command() -> ShellCommand:
    return TagCommand()