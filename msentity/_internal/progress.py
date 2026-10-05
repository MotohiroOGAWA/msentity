from __future__ import annotations

import sys
import time
from typing import TextIO


class SimpleProgress:
    """Small terminal progress indicator with no third-party dependencies."""

    def __init__(
        self,
        *,
        total: int,
        description: str,
        unit: str = "item",
        min_interval: float = 0.5,
        stream: TextIO | None = None,
    ) -> None:
        self.total = max(0, int(total))
        self.description = description
        self.unit = unit
        self.min_interval = min_interval
        self.stream = stream if stream is not None else sys.stderr
        self.current = 0
        self.postfix = ""
        self._last_render = 0.0
        self._last_text = ""
        self._closed = False
        self._render(force=True)

    def update(self, amount: int = 1) -> None:
        self.current = min(self.total, self.current + int(amount)) if self.total else self.current + int(amount)
        self._render(force=self.total > 0 and self.current >= self.total)

    def set_postfix_str(self, value: str) -> None:
        self.postfix = value
        self._render()

    def set_postfix(self, values: dict[str, object]) -> None:
        self.set_postfix_str(", ".join(f"{key}:{value}" for key, value in values.items()))

    def close(self) -> None:
        if self._closed:
            return
        self._render(force=True)
        self.stream.write("\n")
        self.stream.flush()
        self._closed = True

    def _render(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_render < self.min_interval:
            return
        self._last_render = now
        if self.total:
            percent = min(100.0, self.current / self.total * 100.0)
            status = f"{percent:5.1f}% {self.current}/{self.total} {self.unit}"
        else:
            status = f"{self.current} {self.unit}"
        suffix = f" · {self.postfix}" if self.postfix else ""
        text = f"{self.description} {status}{suffix}"
        if text == self._last_text:
            return
        padding = " " * max(0, len(self._last_text) - len(text))
        self.stream.write(f"\r{text}{padding}")
        self.stream.flush()
        self._last_text = text
