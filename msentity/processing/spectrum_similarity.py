from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..core.MSDataset import MSDataset


def _as_numpy_index(index: Sequence[int] | np.ndarray) -> np.ndarray:
    """Convert spectrum indices to a one-dimensional numpy int64 array."""
    index = np.asarray(index, dtype=np.int64)

    if index.ndim != 1:
        raise ValueError("index must be one-dimensional")

    return index


def cosine_similarity_pair(
    ds1: MSDataset,
    index1: Sequence[int] | np.ndarray,
    ds2: MSDataset,
    index2: Sequence[int] | np.ndarray,
    *,
    bin_width: float = 0.01,
    intensity_exponent: float = 1.0,
    max_cum_peaks: int = 200_000,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Compute paired binned cosine similarity.

    This function computes cosine similarity for paired spectra:

        ds1[index1[i]] vs ds2[index2[i]]

    Parameters
    ----------
    ds1:
        First MS dataset.
    index1:
        Spectrum indices for ds1.
    ds2:
        Second MS dataset.
    index2:
        Spectrum indices for ds2.
    bin_width:
        Width of the m/z bin.
    intensity_exponent:
        Exponent applied to non-negative intensities.
        For example, 0.5 means square-root transformed intensities.
    max_cum_peaks:
        Maximum cumulative number of peaks in each internal chunk.
    show_progress:
        If True, show progress bar.

    Returns
    -------
    numpy.ndarray
        Cosine similarity scores with shape ``(K,)``.
    """
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    if intensity_exponent <= 0:
        raise ValueError("intensity_exponent must be positive")

    if max_cum_peaks <= 0:
        raise ValueError("max_cum_peaks must be positive")

    index1 = _as_numpy_index(index1)
    index2 = _as_numpy_index(index2)

    if index1.shape != index2.shape:
        raise ValueError(
            f"index1 and index2 must have the same shape, "
            f"got {index1.shape} and {index2.shape}"
        )

    k = index1.size
    if k == 0:
        return np.empty(0, dtype=np.float32)

    len1 = np.asarray(ds1.peaks.lengths, dtype=np.int64)[index1]
    len2 = np.asarray(ds2.peaks.lengths, dtype=np.int64)[index2]

    scores = np.empty(k, dtype=np.float32)

    start = 0
    pbar = tqdm(total=k, desc="Computing paired cosine similarity") if show_progress else None

    while start < k:
        cumsum1 = np.cumsum(len1[start:])
        cumsum2 = np.cumsum(len2[start:])

        exceed = (cumsum1 > max_cum_peaks) | (cumsum2 > max_cum_peaks)

        if np.any(exceed):
            first_exceed = int(np.flatnonzero(exceed)[0])
            end = start + max(1, first_exceed)
        else:
            end = k

        scores[start:end] = _cosine_similarity_pair_core(
            ds1=ds1,
            index1=index1[start:end],
            ds2=ds2,
            index2=index2[start:end],
            bin_width=bin_width,
            intensity_exponent=intensity_exponent,
        )

        if pbar is not None:
            pbar.update(end - start)

        start = end

    if pbar is not None:
        pbar.close()

    return scores

def cosine_similarity_by_key(
    ds1: MSDataset,
    ds2: MSDataset,
    *,
    key1: str = "SpecID",
    key2: str = "SpecID",
    bin_width: float = 0.01,
    intensity_exponent: float = 1.0,
    max_cum_peaks: int = 200_000,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Compute paired cosine similarity by matching spectrum metadata keys.

    Rows in ``ds1`` and ``ds2`` are paired when ``ds1[key1] == ds2[key2]``.
    Only keys present in both datasets are compared. Key values must be unique
    within each dataset; duplicate keys are rejected to avoid ambiguous
    many-to-many pairings. Missing key values are ignored.

    Parameters
    ----------
    ds1:
        First MS dataset.
    ds2:
        Second MS dataset.
    key1:
        Metadata column in ``ds1`` used as the matching key.
    key2:
        Metadata column in ``ds2`` used as the matching key.
    bin_width:
        Width of the m/z bin.
    intensity_exponent:
        Exponent applied to non-negative intensities.
    max_cum_peaks:
        Maximum cumulative number of peaks in each internal chunk.
    show_progress:
        If True, show progress bar while computing cosine similarity.

    Returns
    -------
    pandas.DataFrame
        Table containing the matched key values, ``index1``, ``index2``, and
        ``cosine_similarity``. The row order follows ``ds1``.

    Raises
    ------
    KeyError
        If ``key1`` or ``key2`` is not available in the corresponding dataset.
    ValueError
        If either key column contains duplicate non-missing values.
    """
    if key1 not in ds1.columns:
        raise KeyError(f"ds1 does not contain key column: {key1}")
    if key2 not in ds2.columns:
        raise KeyError(f"ds2 does not contain key column: {key2}")

    left = pd.DataFrame(
        {
            "index1": np.arange(ds1.n_rows, dtype=np.int64),
            key1: ds1[key1].to_numpy(),
        }
    )
    right = pd.DataFrame(
        {
            "index2": np.arange(ds2.n_rows, dtype=np.int64),
            key2: ds2[key2].to_numpy(),
        }
    )

    # Missing values should not be treated as matching identifiers.
    left = left[left[key1].notna()]
    right = right[right[key2].notna()]

    duplicate1 = left[key1].duplicated(keep=False)
    if duplicate1.any():
        values = left.loc[duplicate1, key1].drop_duplicates().tolist()[:5]
        raise ValueError(
            f"ds1 key column {key1!r} contains duplicate values; "
            f"examples: {values}"
        )

    duplicate2 = right[key2].duplicated(keep=False)
    if duplicate2.any():
        values = right.loc[duplicate2, key2].drop_duplicates().tolist()[:5]
        raise ValueError(
            f"ds2 key column {key2!r} contains duplicate values; "
            f"examples: {values}"
        )

    pairs = left.merge(
        right,
        left_on=key1,
        right_on=key2,
        how="inner",
        sort=False,
        validate="one_to_one",
    )

    scores = cosine_similarity_pair(
        ds1=ds1,
        index1=pairs["index1"].to_numpy(dtype=np.int64),
        ds2=ds2,
        index2=pairs["index2"].to_numpy(dtype=np.int64),
        bin_width=bin_width,
        intensity_exponent=intensity_exponent,
        max_cum_peaks=max_cum_peaks,
        show_progress=show_progress,
    )

    pairs["cosine_similarity"] = scores

    if key1 == key2:
        columns = [key1, "index1", "index2", "cosine_similarity"]
    else:
        columns = [key1, key2, "index1", "index2", "cosine_similarity"]

    return pairs[columns].reset_index(drop=True)

def _aggregate_binned_peaks(
    *,
    mz: np.ndarray,
    intensity: np.ndarray,
    lengths: np.ndarray,
    bin_width: float,
    min_bin: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate peaks into sparse binned vectors.

    Returns
    -------
    keys:
        Flattened sparse keys: ``spectrum_id * n_bins + bin_id``.
    values:
        Summed intensity values for each key.
    spectrum_ids:
        Spectrum id for each key.
    """
    if mz.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    spectrum_ids_per_peak = np.repeat(
        np.arange(lengths.size, dtype=np.int64),
        lengths,
    )

    bins = np.floor(mz / bin_width).astype(np.int64) - min_bin
    keys_per_peak = spectrum_ids_per_peak * n_bins + bins

    keys, inverse = np.unique(keys_per_peak, return_inverse=True)

    values = np.zeros(keys.size, dtype=np.float64)
    np.add.at(values, inverse, intensity)

    spectrum_ids = keys // n_bins

    return keys, values, spectrum_ids


def _cosine_similarity_pair_core(
    *,
    ds1: MSDataset,
    index1: np.ndarray,
    ds2: MSDataset,
    index2: np.ndarray,
    bin_width: float,
    intensity_exponent: float,
) -> np.ndarray:
    """
    Compute paired binned cosine similarity for one chunk.

    This function uses sparse binned vectors internally.
    It avoids creating a dense ``(K, n_bins)`` matrix.
    """
    if index1.ndim != 1 or index2.ndim != 1:
        raise ValueError("index1 and index2 must be one-dimensional")

    if index1.size != index2.size:
        raise ValueError("index1 and index2 must have the same length")

    k = index1.size
    if k == 0:
        return np.empty(0, dtype=np.float32)

    subset1 = ds1[index1]
    subset2 = ds2[index2]

    ps1 = subset1.peaks
    ps2 = subset2.peaks

    mz1 = np.asarray(ps1.mz, dtype=np.float64)
    mz2 = np.asarray(ps2.mz, dtype=np.float64)

    intensity1 = np.asarray(ps1.intensity, dtype=np.float64)
    intensity2 = np.asarray(ps2.intensity, dtype=np.float64)

    lengths1 = np.asarray(ps1.lengths, dtype=np.int64)
    lengths2 = np.asarray(ps2.lengths, dtype=np.int64)

    intensity1 = np.clip(intensity1, a_min=0.0, a_max=None)
    intensity2 = np.clip(intensity2, a_min=0.0, a_max=None)

    if intensity_exponent != 1.0:
        intensity1 = intensity1 ** intensity_exponent
        intensity2 = intensity2 ** intensity_exponent

    if mz1.size == 0 and mz2.size == 0:
        return np.zeros(k, dtype=np.float32)

    if mz1.size == 0:
        bins2 = np.floor(mz2 / bin_width).astype(np.int64)
        min_bin = int(bins2.min())
        max_bin = int(bins2.max())
    elif mz2.size == 0:
        bins1 = np.floor(mz1 / bin_width).astype(np.int64)
        min_bin = int(bins1.min())
        max_bin = int(bins1.max())
    else:
        bins1 = np.floor(mz1 / bin_width).astype(np.int64)
        bins2 = np.floor(mz2 / bin_width).astype(np.int64)
        min_bin = int(min(bins1.min(), bins2.min()))
        max_bin = int(max(bins1.max(), bins2.max()))

    n_bins = max_bin - min_bin + 1

    keys1, values1, spectrum_ids1 = _aggregate_binned_peaks(
        mz=mz1,
        intensity=intensity1,
        lengths=lengths1,
        bin_width=bin_width,
        min_bin=min_bin,
        n_bins=n_bins,
    )

    keys2, values2, spectrum_ids2 = _aggregate_binned_peaks(
        mz=mz2,
        intensity=intensity2,
        lengths=lengths2,
        bin_width=bin_width,
        min_bin=min_bin,
        n_bins=n_bins,
    )

    norm1 = np.zeros(k, dtype=np.float64)
    norm2 = np.zeros(k, dtype=np.float64)

    if values1.size > 0:
        np.add.at(norm1, spectrum_ids1, values1 * values1)

    if values2.size > 0:
        np.add.at(norm2, spectrum_ids2, values2 * values2)

    dot = np.zeros(k, dtype=np.float64)

    if keys1.size > 0 and keys2.size > 0:
        # keys are sorted because np.unique returns sorted values.
        pos = np.searchsorted(keys2, keys1)
        valid = pos < keys2.size

        if np.any(valid):
            keys1_valid = keys1[valid]
            pos_valid = pos[valid]

            matched = keys2[pos_valid] == keys1_valid

            if np.any(matched):
                left_indices = np.flatnonzero(valid)[matched]
                right_indices = pos_valid[matched]

                matched_keys = keys1[left_indices]
                matched_spectrum_ids = matched_keys // n_bins

                dot_values = values1[left_indices] * values2[right_indices]
                np.add.at(dot, matched_spectrum_ids, dot_values)

    denom = np.sqrt(norm1 * norm2)

    score = np.zeros(k, dtype=np.float64)
    np.divide(dot, denom, out=score, where=denom > 0)

    score = np.clip(score, 0.0, 1.0)

    return score.astype(np.float32)


def cosine_similarity_all_pairs_matrix(
    ds1: MSDataset,
    ds2: MSDataset,
    *,
    bin_width: float = 0.01,
    intensity_exponent: float = 1.0,
    max_pairs_per_call: int = 2_000_000,
    max_cum_peaks: int = 200_000,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Compute full all-pairs cosine similarity matrix.

    The output shape is:

        ``(len(ds1), len(ds2))``

    This function does not materialize all pair indices at once.
    It processes flattened pair indices in chunks.
    """
    if max_pairs_per_call <= 0:
        raise ValueError("max_pairs_per_call must be positive")

    n1 = int(ds1.n_rows)
    n2 = int(ds2.n_rows)

    if n1 == 0 or n2 == 0:
        return np.empty((n1, n2), dtype=np.float32)

    similarity = np.empty((n1, n2), dtype=np.float32)
    flat_similarity = similarity.reshape(-1)

    total_pairs = n1 * n2

    start = 0
    pbar = tqdm(total=total_pairs, desc="Computing all-pairs cosine similarity") if show_progress else None

    while start < total_pairs:
        end = min(total_pairs, start + max_pairs_per_call)

        pair_indices = np.arange(start, end, dtype=np.int64)

        index1 = pair_indices // n2
        index2 = pair_indices % n2

        scores = cosine_similarity_pair(
            ds1=ds1,
            index1=index1,
            ds2=ds2,
            index2=index2,
            bin_width=bin_width,
            intensity_exponent=intensity_exponent,
            max_cum_peaks=max_cum_peaks,
            show_progress=False,
        )

        flat_similarity[start:end] = scores

        if pbar is not None:
            pbar.update(end - start)

        start = end

    if pbar is not None:
        pbar.close()

    return similarity