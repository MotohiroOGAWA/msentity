"""Compatibility imports for the former similarity module location."""

from ..similarity.calculation import (
    cosine_similarity_all_pairs_matrix,
    cosine_similarity_by_key,
    cosine_similarity_pair,
)

__all__ = [
    "cosine_similarity_all_pairs_matrix",
    "cosine_similarity_by_key",
    "cosine_similarity_pair",
]
