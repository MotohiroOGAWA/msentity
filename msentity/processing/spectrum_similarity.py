from __future__ import annotations
from typing import Optional, Sequence, Tuple
import torch
from tqdm import tqdm

from ..core.MSDataset import MSDataset


def cosine_similarity_pair(
    ds1,
    index1: torch.Tensor,
    ds2,
    index2: torch.Tensor,
    *,
    bin_width: float = 0.01,
    intensity_exponent: float = 1.0,
    max_cum_peaks: int = 200_000,
    device: Optional[torch.device] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Paired cosine similarity computed in chunks.

    This function splits the (index1, index2) pairs into chunks so that the
    cumulative number of peaks in *either* ds1 or ds2 does not exceed max_cum_peaks.
    Each chunk is computed by _cosine_similarity_pair_core and then concatenated
    back into a single [K] tensor.
    """
    assert index1.ndim == 1 and index2.ndim == 1
    assert index1.numel() == index2.numel()

    K = index1.numel()
    if K == 0:
        return torch.empty(0, device=device or torch.device("cpu"))

    if device is None:
        device = ds1.peaks.device

    ds1 = ds1.to(device=device)
    ds2 = ds2.to(device=device)

    # Peak counts per paired spectrum (on CPU for cheap cumsum)
    len1 = ds1[index1].peaks.length.to("cpu", dtype=torch.long)
    len2 = ds2[index2].peaks.length.to("cpu", dtype=torch.long)

    out = torch.empty(K, device=device, dtype=torch.float32)

    start = 0
    if show_progress:
        pbar = tqdm(total=K, desc="Computing paired cosine similarity")
    else:
        pbar = None

    while start < K:
        # Build cumulative peak counts starting from `start`
        c1 = torch.cumsum(len1[start:], dim=0)
        c2 = torch.cumsum(len2[start:], dim=0)

        # Find the first position where either cumulative sum exceeds the budget
        exceed = (c1 > max_cum_peaks) | (c2 > max_cum_peaks)

        if torch.any(exceed):
            # first_exceed is the offset (>=0) where it first becomes True
            first_exceed = int(torch.nonzero(exceed, as_tuple=False)[0].item())
            # Use the range that stays within budget; ensure at least one pair
            end = start + max(1, first_exceed)
        else:
            end = K

        # Compute this chunk using the core routine (no chunking inside)
        sim_chunk = _cosine_similarity_pair_core(
            ds1,
            index1[start:end],
            ds2,
            index2[start:end],
            bin_width=bin_width,
            intensity_exponent=intensity_exponent,
            device=device,
        )
        out[start:end] = sim_chunk

        if pbar is not None:
            pbar.update(end - start)
            
        start = end


    if pbar is not None:
        pbar.close()

    return out

def _cosine_similarity_pair_core(
    ds1,
    index1: torch.Tensor,
    ds2,
    index2: torch.Tensor,
    *,
    bin_width: float,
    intensity_exponent: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Paired cosine similarity using m/z binning.

    Notes
    -----
    - Each spectrum pair is compared in a shared binned m/z space.
    - Peaks falling into the same bin are summed before cosine calculation.
    - Scores are clamped to [0, 1] to remove tiny numerical overflow.
    """
    assert index1.ndim == 1 and index2.ndim == 1
    assert index1.numel() == index2.numel()
    assert bin_width > 0

    subset1 = ds1[index1]
    subset2 = ds2[index2]

    ps1 = subset1.peaks
    ps2 = subset2.peaks

    K = index1.numel()
    if K == 0:
        return torch.empty(0, device=device, dtype=torch.float32)

    # -------------------------
    # Flatten all peaks
    # -------------------------
    mz1 = ps1.mz.to(device=device, dtype=torch.float64)
    it1 = ps1.intensity.to(device=device, dtype=torch.float64)
    mz2 = ps2.mz.to(device=device, dtype=torch.float64)
    it2 = ps2.intensity.to(device=device, dtype=torch.float64)

    # Intensity transform
    it1 = torch.clamp(it1, min=0.0)
    it2 = torch.clamp(it2, min=0.0)
    if intensity_exponent != 1.0:
        it1 = it1.pow(intensity_exponent)
        it2 = it2.pow(intensity_exponent)

    len1 = ps1.length.to(device=device, dtype=torch.long)
    len2 = ps2.length.to(device=device, dtype=torch.long)

    spec_id1 = torch.repeat_interleave(
        torch.arange(K, device=device, dtype=torch.long),
        len1,
    )
    spec_id2 = torch.repeat_interleave(
        torch.arange(K, device=device, dtype=torch.long),
        len2,
    )

    # -------------------------
    # Convert m/z to integer bins
    # -------------------------
    bin1 = torch.floor(mz1 / bin_width).to(torch.long)
    bin2 = torch.floor(mz2 / bin_width).to(torch.long)

    if bin1.numel() == 0 and bin2.numel() == 0:
        return torch.zeros(K, device=device, dtype=torch.float32)

    if bin1.numel() == 0:
        min_bin = bin2.min()
        max_bin = bin2.max()
    elif bin2.numel() == 0:
        min_bin = bin1.min()
        max_bin = bin1.max()
    else:
        min_bin = torch.minimum(bin1.min(), bin2.min())
        max_bin = torch.maximum(bin1.max(), bin2.max())

    n_bins = int((max_bin - min_bin + 1).item())

    bin1 = bin1 - min_bin
    bin2 = bin2 - min_bin

    # -------------------------
    # Build binned vectors
    # flat_index = spec_id * n_bins + bin_id
    # -------------------------
    flat_size = K * n_bins
    vec1 = torch.zeros(flat_size, device=device, dtype=torch.float64)
    vec2 = torch.zeros(flat_size, device=device, dtype=torch.float64)

    if bin1.numel() > 0:
        flat_idx1 = spec_id1 * n_bins + bin1
        vec1.scatter_add_(0, flat_idx1, it1)

    if bin2.numel() > 0:
        flat_idx2 = spec_id2 * n_bins + bin2
        vec2.scatter_add_(0, flat_idx2, it2)

    vec1 = vec1.view(K, n_bins)
    vec2 = vec2.view(K, n_bins)

    # -------------------------
    # Cosine similarity
    # -------------------------
    dot = (vec1 * vec2).sum(dim=1)
    norm1 = (vec1 * vec1).sum(dim=1)
    norm2 = (vec2 * vec2).sum(dim=1)

    denom = torch.sqrt(norm1 * norm2).clamp(min=1e-12)
    score = dot / denom

    # Remove tiny numerical overflow
    score = torch.clamp(score, min=0.0, max=1.0)

    return score.to(torch.float32)

# def _cosine_similarity_pair_core(
#     ds1,
#     index1: torch.Tensor,
#     ds2,
#     index2: torch.Tensor,
#     *,
#     bin_width: float,
#     intensity_exponent: float,
#     device: torch.device,
# ) -> torch.Tensor:
#     """
#     Paired cosine similarity using m/z binning.

#     Notes
#     -----
#     - Each spectrum pair is compared in a shared binned m/z space.
#     - This guarantees the cosine score is <= 1 up to numerical error.
#     - Peaks falling into the same bin are summed.
#     """
#     assert index1.ndim == 1 and index2.ndim == 1
#     assert index1.numel() == index2.numel()
#     assert bin_width > 0

#     subset1 = ds1[index1]
#     subset2 = ds2[index2]

#     ps1 = subset1.peaks
#     ps2 = subset2.peaks

#     K = index1.numel()
#     if K == 0:
#         return torch.empty(0, device=device)

#     # Flatten all peaks
#     mz1 = ps1.mz.to(device).float()
#     it1 = ps1.intensity.to(device).float()
#     mz2 = ps2.mz.to(device).float()
#     it2 = ps2.intensity.to(device).float()

#     # Intensity power transform
#     if intensity_exponent != 1.0:
#         it1 = torch.clamp(it1, min=0.0) ** intensity_exponent
#         it2 = torch.clamp(it2, min=0.0) ** intensity_exponent
#     else:
#         it1 = torch.clamp(it1, min=0.0)
#         it2 = torch.clamp(it2, min=0.0)

#     # Spectrum id per peak within the chunk
#     spec_id1 = torch.repeat_interleave(
#         torch.arange(K, device=device, dtype=torch.long),
#         ps1.length.to(device),
#     )
#     spec_id2 = torch.repeat_interleave(
#         torch.arange(K, device=device, dtype=torch.long),
#         ps2.length.to(device),
#     )

#     # ---------------------------------------------------------
#     # Convert each peak to an integer bin id
#     # ---------------------------------------------------------
#     bin1 = torch.floor(mz1 / bin_width).long()
#     bin2 = torch.floor(mz2 / bin_width).long()

#     if bin1.numel() == 0 and bin2.numel() == 0:
#         return torch.zeros(K, device=device)

#     # Shared global bin range within this chunk
#     if bin1.numel() == 0:
#         min_bin = bin2.min()
#         max_bin = bin2.max()
#     elif bin2.numel() == 0:
#         min_bin = bin1.min()
#         max_bin = bin1.max()
#     else:
#         min_bin = torch.minimum(bin1.min(), bin2.min())
#         max_bin = torch.maximum(bin1.max(), bin2.max())

#     n_bins = int((max_bin - min_bin + 1).item())

#     # Shift to [0, n_bins)
#     bin1 = bin1 - min_bin
#     bin2 = bin2 - min_bin

#     # ---------------------------------------------------------
#     # Build per-spectrum binned vectors using flattened indexing
#     # flat_index = spec_id * n_bins + bin_id
#     # ---------------------------------------------------------
#     flat_size = K * n_bins

#     vec1 = torch.zeros(flat_size, device=device, dtype=it1.dtype)
#     vec2 = torch.zeros(flat_size, device=device, dtype=it2.dtype)

#     if bin1.numel() > 0:
#         flat_idx1 = spec_id1 * n_bins + bin1
#         vec1.scatter_add_(0, flat_idx1, it1)

#     if bin2.numel() > 0:
#         flat_idx2 = spec_id2 * n_bins + bin2
#         vec2.scatter_add_(0, flat_idx2, it2)

#     vec1 = vec1.view(K, n_bins)
#     vec2 = vec2.view(K, n_bins)

#     # ---------------------------------------------------------
#     # Cosine similarity
#     # ---------------------------------------------------------
#     dot = (vec1 * vec2).sum(dim=1)
#     norm1 = (vec1 * vec1).sum(dim=1)
#     norm2 = (vec2 * vec2).sum(dim=1)

#     denom = torch.sqrt(norm1 * norm2).clamp(min=1e-12)
#     score = dot / denom

#     return score

def cosine_similarity_all_pairs_matrix(
    ds1: MSDataset,
    ds2: MSDataset,
    *,
    bin_width: float = 0.01,
    intensity_exponent: float = 1.0,
    device: Optional[torch.device] = None,
    max_pairs_per_call: int = 2_000_000,
) -> torch.Tensor:
    """
    Compute full N1 x N2 cosine similarity matrix by calling the paired function:

        cosine_similarity_matrix(ds1, index1, ds2, index2)

    This implementation is memory-safe for huge n2 because it NEVER materializes
    the full idx1/idx2 arrays of length n1*n2. Instead, it generates idx1/idx2
    per chunk from flattened pair indices p.

    Returns:
        Tensor of shape [N1, N2]
    """

    # Determine number of spectra in each dataset
    n1 = int(ds1.n_rows)
    n2 = int(ds2.n_rows)

    # Return empty matrix if one dataset is empty
    if n1 == 0 or n2 == 0:
        dev = device if device is not None else ds1.peaks.device
        return torch.empty((n1, n2), dtype=torch.float32, device=dev)

    dev = device if device is not None else ds1.peaks.device

    # Allocate output similarity matrix
    S = torch.empty((n1, n2), dtype=torch.float32, device=dev)

    total_pairs = n1 * n2
    if max_pairs_per_call <= 0:
        raise ValueError("max_pairs_per_call must be a positive integer")

    start = 0
    flat_S = S.view(-1)

    while start < total_pairs:
        end = min(total_pairs, start + max_pairs_per_call)

        # Build flattened pair indices p in [start, end)
        p = torch.arange(start, end, device=dev, dtype=torch.int64)

        # Convert flattened index p -> (i, j), where p = i * n2 + j
        idx1 = torch.div(p, n2, rounding_mode="floor")
        idx2 = torch.remainder(p, n2)

        sims = cosine_similarity_pair(
            ds1, idx1,
            ds2, idx2,
            bin_width=bin_width,
            intensity_exponent=intensity_exponent,
            device=dev,
        )

        # Scatter computed similarities into the flattened output matrix
        flat_S[p] = sims

        start = end

    return S