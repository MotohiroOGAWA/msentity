from __future__ import annotations

import atexit
from pathlib import Path
from typing import Iterable


def setup_history(
    *,
    history_file: str | Path | None = None,
    max_length: int = 1000,
    completions: Iterable[str] | None = None,
) -> None:
    """Enable command history and simple tab completion."""
    try:
        import readline
    except ImportError:
        return

    if history_file is None:
        history_file = Path.home() / ".msentity_history"
    else:
        history_file = Path(history_file)

    try:
        history_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError:
        return

    try:
        readline.read_history_file(str(history_file))
    except FileNotFoundError:
        pass
    except OSError:
        pass

    try:
        readline.set_history_length(max_length)
    except Exception:
        pass

    if completions is not None:
        setup_completion(
            readline_module=readline,
            completions=list(completions),
        )

    def save_history() -> None:
        try:
            readline.write_history_file(str(history_file))
        except OSError:
            pass

    atexit.register(save_history)


def setup_completion(
    *,
    readline_module,
    completions: list[str],
) -> None:
    words = sorted(set(completions))

    def complete(
        text: str,
        state: int,
    ) -> str | None:
        matches = [
            word for word in words
            if word.startswith(text)
        ]

        if state < len(matches):
            return matches[state]

        return None

    readline_module.set_completer(complete)
    readline_module.parse_and_bind("tab: complete")