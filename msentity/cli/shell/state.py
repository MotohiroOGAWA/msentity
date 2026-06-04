from __future__ import annotations

from dataclasses import dataclass

from msentity import MSDataset


@dataclass
class ShellState:
    """State of the dataset shell."""

    dataset: MSDataset
    input_file: str