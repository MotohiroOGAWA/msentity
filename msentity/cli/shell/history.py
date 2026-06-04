from __future__ import annotations

import atexit
from pathlib import Path


def setup_history(
    history_file: str | Path | None = None,
    *,
    max_length: int = 1000,
) -> None:
    """
    Enable command history for the shell.

    Notes
    -----
    This function uses the standard-library readline module.
    On some platforms, readline may be unavailable.
    In that case, history is silently disabled.
    """
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

    def save_history() -> None:
        try:
            readline.write_history_file(str(history_file))
        except OSError:
            pass

    atexit.register(save_history)