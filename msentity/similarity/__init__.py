"""Spectrum similarity calculations and persistent result datasets."""

from .calculation import (
    cosine_similarity_all_pairs_matrix,
    cosine_similarity_by_key,
    cosine_similarity_pair,
    library_search,
    similarity_by_key,
)
from .SimilarityDataset import (
    SimilarityDataset,
    SimilarityResult,
    calculate_library_search,
    calculate_similarity,
)

__all__ = [
    "SimilarityDataset",
    "SimilarityResult",
    "calculate_library_search",
    "calculate_similarity",
    "cosine_similarity_all_pairs_matrix",
    "cosine_similarity_by_key",
    "cosine_similarity_pair",
    "library_search",
    "similarity_by_key",
]
