import unittest

import numpy as np
import pandas as pd

from msentity.core.MSDataset import MSDataset
from msentity.core.PeakSeries import PeakSeries
from msentity.processing.spectrum_similarity import (
    cosine_similarity_all_pairs_matrix,
    cosine_similarity_pair,
)


class TestSpectrumSimilarityFunctions(unittest.TestCase):
    """Unit tests for spectrum similarity utilities."""

    def setUp(self) -> None:
        self.dataset = self._make_dataset()

    def _make_dataset(self) -> MSDataset:
        """Create a small MSDataset for spectrum similarity tests."""
        peak_data = np.array(
            [
                # spectrum 0
                [100.00, 1.0],
                [101.00, 2.0],
                [102.00, 3.0],
                # spectrum 1: identical to spectrum 0
                [100.00, 1.0],
                [101.00, 2.0],
                [102.00, 3.0],
                # spectrum 2: partially overlapping with spectrum 0
                [100.00, 1.0],
                [105.00, 2.0],
                # spectrum 3: no overlapping bins with spectrum 0
                [200.00, 1.0],
                [201.00, 2.0],
                # spectrum 4: empty spectrum
            ],
            dtype=float,
        )

        offsets = np.array([0, 3, 6, 8, 10, 10], dtype=np.int64)

        spectrum_metadata = pd.DataFrame(
            {
                "name": ["spec0", "spec1", "spec2", "spec3", "empty"],
                "precursor_mz": [150.0, 150.0, 155.0, 250.0, 0.0],
            }
        )

        peak_series = PeakSeries(
            data=peak_data,
            offsets=offsets,
        )

        return MSDataset(
            spectrum_metadata=spectrum_metadata,
            peak_series=peak_series,
            columns=["name", "precursor_mz"],
            description="spectrum similarity test dataset",
            attributes={"source": "unit-test"},
            tags=["test"],
        )

    def test_cosine_similarity_pair_identical_spectra(self) -> None:
        """Identical spectra should have cosine similarity 1."""
        scores = cosine_similarity_pair(
            self.dataset,
            np.array([0]),
            self.dataset,
            np.array([1]),
            bin_width=0.01,
        )

        self.assertEqual(scores.shape, (1,))
        self.assertEqual(scores.dtype, np.float32)
        self.assertAlmostEqual(float(scores[0]), 1.0, places=6)

    def test_cosine_similarity_pair_no_overlap(self) -> None:
        """Spectra without overlapping bins should have cosine similarity 0."""
        scores = cosine_similarity_pair(
            self.dataset,
            np.array([0]),
            self.dataset,
            np.array([3]),
            bin_width=0.01,
        )

        self.assertAlmostEqual(float(scores[0]), 0.0, places=6)

    def test_cosine_similarity_pair_partial_overlap(self) -> None:
        """Partially overlapping spectra should return the expected cosine score."""
        scores = cosine_similarity_pair(
            self.dataset,
            np.array([0]),
            self.dataset,
            np.array([2]),
            bin_width=0.01,
        )

        # spec0 = [100:1, 101:2, 102:3]
        # spec2 = [100:1, 105:2]
        # dot = 1*1 = 1
        # norm0 = sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
        # norm2 = sqrt(1^2 + 2^2) = sqrt(5)
        expected = 1.0 / np.sqrt(14.0 * 5.0)

        self.assertAlmostEqual(float(scores[0]), float(expected), places=6)

    def test_cosine_similarity_pair_multiple_pairs(self) -> None:
        """Multiple paired comparisons should be computed in input order."""
        scores = cosine_similarity_pair(
            self.dataset,
            np.array([0, 0, 0]),
            self.dataset,
            np.array([1, 2, 3]),
            bin_width=0.01,
        )

        expected_partial = 1.0 / np.sqrt(14.0 * 5.0)

        self.assertEqual(scores.shape, (3,))
        self.assertAlmostEqual(float(scores[0]), 1.0, places=6)
        self.assertAlmostEqual(float(scores[1]), float(expected_partial), places=6)
        self.assertAlmostEqual(float(scores[2]), 0.0, places=6)

    def test_cosine_similarity_pair_empty_index(self) -> None:
        """Empty input indices should return an empty score array."""
        scores = cosine_similarity_pair(
            self.dataset,
            np.array([], dtype=np.int64),
            self.dataset,
            np.array([], dtype=np.int64),
        )

        self.assertEqual(scores.shape, (0,))
        self.assertEqual(scores.dtype, np.float32)

    def test_cosine_similarity_pair_empty_spectrum(self) -> None:
        """Comparison with an empty spectrum should return 0."""
        scores = cosine_similarity_pair(
            self.dataset,
            np.array([0, 4]),
            self.dataset,
            np.array([4, 4]),
            bin_width=0.01,
        )

        self.assertEqual(scores.shape, (2,))
        self.assertAlmostEqual(float(scores[0]), 0.0, places=6)
        self.assertAlmostEqual(float(scores[1]), 0.0, places=6)

    def test_cosine_similarity_pair_chunking(self) -> None:
        """Chunked computation should produce the same result as non-chunked computation."""
        index1 = np.array([0, 0, 0, 1, 2, 3, 4], dtype=np.int64)
        index2 = np.array([1, 2, 3, 0, 0, 0, 0], dtype=np.int64)

        scores_chunked = cosine_similarity_pair(
            self.dataset,
            index1,
            self.dataset,
            index2,
            bin_width=0.01,
            max_cum_peaks=3,
        )

        scores_non_chunked = cosine_similarity_pair(
            self.dataset,
            index1,
            self.dataset,
            index2,
            bin_width=0.01,
            max_cum_peaks=1_000_000,
        )

        np.testing.assert_allclose(
            scores_chunked,
            scores_non_chunked,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_cosine_similarity_pair_intensity_exponent(self) -> None:
        """intensity_exponent should transform intensities before cosine calculation."""
        scores = cosine_similarity_pair(
            self.dataset,
            np.array([0]),
            self.dataset,
            np.array([2]),
            bin_width=0.01,
            intensity_exponent=0.5,
        )

        # spec0 after sqrt transform = [sqrt(1), sqrt(2), sqrt(3)]
        # spec2 after sqrt transform = [sqrt(1), sqrt(2)]
        # dot = 1
        # norm0^2 = 1 + 2 + 3 = 6
        # norm2^2 = 1 + 2 = 3
        expected = 1.0 / np.sqrt(6.0 * 3.0)

        self.assertAlmostEqual(float(scores[0]), float(expected), places=6)

    def test_cosine_similarity_pair_bins_close_mz_values(self) -> None:
        """Peaks in the same m/z bin should be treated as matching peaks."""
        dataset = self._make_close_mz_dataset()

        scores = cosine_similarity_pair(
            dataset,
            np.array([0]),
            dataset,
            np.array([1]),
            bin_width=0.01,
        )

        self.assertAlmostEqual(float(scores[0]), 1.0, places=6)

    def _make_close_mz_dataset(self) -> MSDataset:
        """Create a dataset with close m/z values that fall into the same bin."""
        peak_data = np.array(
            [
                [100.001, 10.0],
                [100.004, 20.0],
            ],
            dtype=float,
        )

        offsets = np.array([0, 1, 2], dtype=np.int64)

        spectrum_metadata = pd.DataFrame(
            {
                "name": ["a", "b"],
                "precursor_mz": [150.0, 150.0],
            }
        )

        peak_series = PeakSeries(
            data=peak_data,
            offsets=offsets,
        )

        return MSDataset(
            spectrum_metadata=spectrum_metadata,
            peak_series=peak_series,
            columns=["name", "precursor_mz"],
        )

    def test_cosine_similarity_pair_invalid_index_shape(self) -> None:
        """Non-one-dimensional indices should raise ValueError."""
        with self.assertRaises(ValueError):
            cosine_similarity_pair(
                self.dataset,
                np.array([[0, 1]]),
                self.dataset,
                np.array([0, 1]),
            )

    def test_cosine_similarity_pair_mismatched_index_length(self) -> None:
        """index1 and index2 must have the same length."""
        with self.assertRaises(ValueError):
            cosine_similarity_pair(
                self.dataset,
                np.array([0, 1]),
                self.dataset,
                np.array([0]),
            )

    def test_cosine_similarity_pair_invalid_bin_width(self) -> None:
        """bin_width must be positive."""
        with self.assertRaises(ValueError):
            cosine_similarity_pair(
                self.dataset,
                np.array([0]),
                self.dataset,
                np.array([1]),
                bin_width=0.0,
            )

    def test_cosine_similarity_pair_invalid_intensity_exponent(self) -> None:
        """intensity_exponent must be positive."""
        with self.assertRaises(ValueError):
            cosine_similarity_pair(
                self.dataset,
                np.array([0]),
                self.dataset,
                np.array([1]),
                intensity_exponent=0.0,
            )

    def test_cosine_similarity_all_pairs_matrix_shape(self) -> None:
        """All-pairs matrix should have shape len(ds1) x len(ds2)."""
        matrix = cosine_similarity_all_pairs_matrix(
            self.dataset,
            self.dataset,
            bin_width=0.01,
        )

        self.assertEqual(matrix.shape, (len(self.dataset), len(self.dataset)))
        self.assertEqual(matrix.dtype, np.float32)

    def test_cosine_similarity_all_pairs_matrix_values(self) -> None:
        """All-pairs matrix should contain expected pairwise cosine scores."""
        matrix = cosine_similarity_all_pairs_matrix(
            self.dataset,
            self.dataset,
            bin_width=0.01,
            max_pairs_per_call=3,
        )

        expected_partial = 1.0 / np.sqrt(14.0 * 5.0)

        self.assertAlmostEqual(float(matrix[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(matrix[0, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(matrix[0, 2]), float(expected_partial), places=6)
        self.assertAlmostEqual(float(matrix[0, 3]), 0.0, places=6)
        self.assertAlmostEqual(float(matrix[0, 4]), 0.0, places=6)

    def test_cosine_similarity_all_pairs_matrix_is_symmetric_for_same_dataset(self) -> None:
        """All-pairs matrix should be symmetric when ds1 and ds2 are the same dataset."""
        matrix = cosine_similarity_all_pairs_matrix(
            self.dataset,
            self.dataset,
            bin_width=0.01,
            max_pairs_per_call=4,
        )

        np.testing.assert_allclose(
            matrix,
            matrix.T,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_cosine_similarity_all_pairs_matrix_empty_dataset(self) -> None:
        """All-pairs matrix should support empty datasets."""
        empty_dataset = self._make_empty_dataset()

        matrix1 = cosine_similarity_all_pairs_matrix(
            empty_dataset,
            self.dataset,
            bin_width=0.01,
        )
        matrix2 = cosine_similarity_all_pairs_matrix(
            self.dataset,
            empty_dataset,
            bin_width=0.01,
        )

        self.assertEqual(matrix1.shape, (0, len(self.dataset)))
        self.assertEqual(matrix2.shape, (len(self.dataset), 0))

    def _make_empty_dataset(self) -> MSDataset:
        """Create an empty MSDataset."""
        peak_data = np.empty((0, 2), dtype=float)
        offsets = np.array([0], dtype=np.int64)

        spectrum_metadata = pd.DataFrame(
            {
                "name": [],
                "precursor_mz": [],
            }
        )

        peak_series = PeakSeries(
            data=peak_data,
            offsets=offsets,
        )

        return MSDataset(
            spectrum_metadata=spectrum_metadata,
            peak_series=peak_series,
            columns=["name", "precursor_mz"],
        )

    def test_cosine_similarity_all_pairs_matrix_invalid_max_pairs_per_call(self) -> None:
        """max_pairs_per_call must be positive."""
        with self.assertRaises(ValueError):
            cosine_similarity_all_pairs_matrix(
                self.dataset,
                self.dataset,
                max_pairs_per_call=0,
            )


if __name__ == "__main__":
    unittest.main()