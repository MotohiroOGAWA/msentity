from __future__ import annotations

from dataclasses import dataclass

from msentity import MSDataset


@dataclass
class ShellState:
    """State of the dataset shell."""

    original_dataset: MSDataset
    dataset: MSDataset
    input_file: str