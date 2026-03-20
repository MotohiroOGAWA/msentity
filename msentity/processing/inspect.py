from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py


def print_hdf5_structure(
    path: Union[str, Path],
    *,
    show_attrs: bool = True,
    show_datasets: bool = True,
    max_depth: int | None = None,
) -> None:
    """
    Print the hierarchical structure of an HDF5 file.

    This function prints groups and datasets in a tree-like order starting
    from the root group. Optionally, it can also print dataset metadata
    such as shape and dtype, as well as HDF5 attributes.

    Parameters
    ----------
    path
        Path to the HDF5 file.
    show_attrs
        Whether to print attributes attached to groups and datasets.
    show_datasets
        Whether to print dataset details such as shape and dtype.
        If ``False``, only the dataset names are shown.
    max_depth
        Maximum depth to traverse from the root group.

        - ``None``: traverse all levels
        - ``0``: show only the root group
        - ``1``: show direct children of the root group
        - and so on

    Returns
    -------
    None
        This function prints the HDF5 structure to standard output.

    Raises
    ------
    TypeError
        If ``path`` is not a string or ``Path`` object.
    ValueError
        If ``max_depth`` is negative.

    Notes
    -----
    - Groups are printed as ``[Group]``.
    - Datasets are printed as ``[Dataset]``.
    - Attributes are printed with the ``@attr`` prefix.
    - Depth is counted from the root group ``/``, which has depth 0.

    Examples
    --------
    Print the full structure:

    >>> print_hdf5_structure("data/example.h5")

    Print only up to depth 1:

    >>> print_hdf5_structure("data/example.h5", max_depth=1)

    Print without attributes:

    >>> print_hdf5_structure("data/example.h5", show_attrs=False)
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path.")
    if max_depth is not None:
        if not isinstance(max_depth, int):
            raise TypeError("max_depth must be an integer or None.")
        if max_depth < 0:
            raise ValueError("max_depth must be greater than or equal to 0.")

    path = Path(path)

    def _print_attrs(obj: h5py.Group | h5py.Dataset, indent: str) -> None:
        """Print attributes of an HDF5 object."""
        if show_attrs and len(obj.attrs) > 0:
            for key, value in obj.attrs.items():
                print(f"{indent}  @attr {key}: {value}")

    def _walk(obj: h5py.Group | h5py.Dataset, name: str, depth: int) -> None:
        """Recursively print an HDF5 object and its children."""
        if max_depth is not None and depth > max_depth:
            return

        indent = "  " * depth

        if isinstance(obj, h5py.Group):
            print(f"{indent}[Group] {name}")
            _print_attrs(obj, indent)

            if max_depth is not None and depth == max_depth:
                return

            for key in obj.keys():
                child = obj[key]
                child_name = f"{name.rstrip('/')}/{key}" if name != "/" else f"/{key}"
                _walk(child, child_name, depth + 1)

        elif isinstance(obj, h5py.Dataset):
            if show_datasets:
                print(
                    f"{indent}[Dataset] {name} "
                    f"shape={obj.shape}, dtype={obj.dtype}"
                )
            else:
                print(f"{indent}[Dataset] {name}")

            _print_attrs(obj, indent)

    with h5py.File(path, "r") as f:
        print(f"HDF5 file: {path}")
        _walk(f, "/", 0)