"""Spectrum similarity calculations and persistent result datasets."""

from .calculation import (
    cosine_similarity_all_pairs_matrix,
    cosine_similarity_by_key,
    cosine_similarity_pair,
)
from .SimilarityDataset import SimilarityDataset, SimilarityResult, calculate_similarity

__all__ = [
    "SimilarityDataset",
    "SimilarityResult",
    "calculate_similarity",
    "cosine_similarity_all_pairs_matrix",
    "cosine_similarity_by_key",
    "cosine_similarity_pair",
]
