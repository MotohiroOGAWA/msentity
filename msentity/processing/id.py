from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.MSDataset import MSDataset


def set_spec_id(
    dataset: MSDataset,
    *,
    col_name: str = "SpecID",
    prefix: str = "",
    overwrite: bool = False,
    start: int = 1,
) -> bool:
    """
    Assign sequential spectrum identifiers to the spectrum metadata table.

    This function adds a new column to ``dataset`` containing one identifier
    for each spectrum. Identifiers are generated sequentially with optional
    zero-padding and prefix.

    For example, if the dataset contains 12 spectra and ``prefix="SP"``,
    the generated identifiers will be::

        SP01, SP02, ..., SP12

    Parameters
    ----------
    dataset
        Target MS dataset.
    col_name
        Name of the metadata column to create.
        For example, ``"SpecID"``, ``"SpectrumID"``, or ``"SampleSpectrumID"``.
    prefix
        String prefix added before each numeric identifier.
    overwrite
        Whether to overwrite an existing column with the same name.
        If ``False`` and the column already exists, the function returns ``False``.
    start
        Starting number for the sequential identifiers.
        The default is ``1``.

    Returns
    -------
    bool
        ``True`` if the identifiers were assigned successfully,
        ``False`` if assignment was skipped because the column already exists
        and ``overwrite=False``.

    Raises
    ------
    TypeError
        If ``col_name`` or ``prefix`` is not a string.
    ValueError
        If ``start`` is not an integer greater than or equal to 0.

    Notes
    -----
    - The numeric part is zero-padded based on the largest identifier.
    - The generated identifiers are stored in the spectrum metadata table.
    - This function writes to ``dataset[col_name]``.

    Examples
    --------
    Create default spectrum IDs:

    >>> set_spec_id(dataset)
    True

    Create spectrum IDs in a custom column:

    >>> set_spec_id(dataset, col_name="SpectrumID", prefix="SP")
    True

    Generated values may look like:

    >>> dataset["SpectrumID"][:3]
    ["SP01", "SP02", "SP03"]
    """
    if not isinstance(col_name, str):
        raise TypeError("col_name must be a string.")
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string.")
    if not isinstance(start, int) or start < 0:
        raise ValueError("start must be an integer greater than or equal to 0.")

    metadata = dataset._spectrum_metadata_ref

    if (col_name in metadata.columns) and (not overwrite):
        print(f"Warning: '{col_name}' column already exists in the dataset.")
        return False

    n = len(dataset)

    if n == 0:
        dataset[col_name] = []
        return True

    # Determine zero-padding width from the largest assigned number.
    last_number = start + n - 1
    width = len(str(last_number))

    spec_ids = [f"{prefix}{i:0{width}d}" for i in range(start, start + n)]
    dataset[col_name] = spec_ids
    return True


def set_peak_id(
    dataset: MSDataset,
    *,
    col_name: str = "PeakID",
    overwrite: bool = False,
    start: int = 0,
) -> bool:
    """
    Assign sequential peak identifiers for each spectrum.

    This function creates a peak-level metadata column in ``dataset.peaks``.
    Identifiers are assigned independently within each spectrum.

    The numbering policy is:

    - For each spectrum, peak identifiers start at ``start``
    - Peak identifiers increase by 1 within that spectrum
    - The sequence resets for the next spectrum

    For example, if the peak counts per spectrum are ``[3, 1, 4]`` and
    ``start=0``, the assigned identifiers will be::

        [0, 1, 2, 0, 0, 1, 2, 3]

    Parameters
    ----------
    dataset
        Target MS dataset.
    col_name
        Name of the peak metadata column to create.
    overwrite
        Whether to overwrite an existing column with the same name.
        If ``False`` and the column already exists, the function returns ``False``.
    start
        Starting number for peak identifiers within each spectrum.

    Returns
    -------
    bool
        ``True`` if the identifiers were assigned successfully,
        ``False`` if assignment was skipped because the column already exists
        and ``overwrite=False``.

    Raises
    ------
    ValueError
        If ``dataset.peaks._offsets_ref`` is not a 1D ``torch.int64`` tensor.

    Notes
    -----
    - Peak identifiers are assigned per spectrum, not globally across the dataset.
    - This function writes to ``dataset.peaks[col_name]``.
    - The output values are stored as strings.

    Examples
    --------
    Create default peak IDs:

    >>> set_peak_id(dataset)
    True

    Create peak IDs starting from 1:

    >>> set_peak_id(dataset, start=1)
    True

    Store them under a custom column:

    >>> set_peak_id(dataset, col_name="LocalPeakID")
    True
    """
    ps = dataset.peaks

    # If peak metadata already exists and the target column is present,
    # skip assignment unless overwriting is explicitly allowed.
    if (
        ps._metadata_ref is not None
        and col_name in ps._metadata_ref.columns
        and not overwrite
    ):
        print(f"Warning: '{col_name}' column already exists in peaks metadata.")
        return False

    # Peak offsets define the start and end positions of peaks for each spectrum.
    # Expected shape: [n_spectra + 1]
    offsets = ps._offsets_ref
    if not isinstance(offsets, np.ndarray) or offsets.ndim != 1 or offsets.dtype != np.int64:
        raise ValueError("dataset.peaks._offsets_ref must be a 1D np.int64 array.")

    # Number of peaks in each spectrum.
    lens = offsets[1:] - offsets[:-1]

    # Total number of peaks across all spectra.
    total = int(lens.sum().item())

    if total == 0:
        ps[col_name] = []
        return True

    # Construct local peak indices efficiently without a Python loop.
    #
    # Example:
    #   offsets = [0, 3, 4, 8]
    #   lens    = [3, 1, 4]
    #
    # Then:
    #   peak_global = [0, 1, 2, 3, 4, 5, 6, 7]
    #   base        = [0, 0, 0, 3, 4, 4, 4, 4]
    #   peak_id     = [0, 1, 2, 0, 0, 1, 2, 3]
    peak_global = np.arange(total, dtype=np.int64)
    base = np.repeat(offsets[:-1], lens)
    peak_id = peak_global - base + int(start)

    # Convert to Python strings before storing in pandas-backed metadata.
    ps[col_name] = [str(v) for v in peak_id]
    return True