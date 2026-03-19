import unittest

import numpy as np
import pandas as pd

from msentity.core.Peak import Peak
from msentity.core.PeakSeries import PeakSeries, Spectrum


class TestPeakSeries(unittest.TestCase):
    """Unit tests for PeakSeries and Spectrum."""

    def setUp(self) -> None:
        self.data = np.array(
            [
                [100.0, 10.0],
                [101.0, 30.0],
                [102.0, 20.0],
                [200.0, 5.0],
                [201.0, 15.0],
                [300.0, 7.0],
            ],
            dtype=float,
        )
        self.offsets = np.array([0, 3, 5, 6], dtype=np.int64)
        self.metadata = pd.DataFrame(
            {
                "annotation": ["a", "b", "c", "d", "e", "f"],
                "group": [1, 1, 1, 2, 2, 3],
            }
        )
        self.series = PeakSeries(
            data=self.data.copy(),
            offsets=self.offsets.copy(),
            metadata=self.metadata.copy(),
            metadata_columns=["annotation", "group"],
        )

    def test_len(self) -> None:
        self.assertEqual(len(self.series), 3)
        self.assertEqual(self.series.count, 3)

    def test_n_peaks_total(self) -> None:
        self.assertEqual(self.series.n_peaks_total, 6)

    def test_lengths(self) -> None:
        np.testing.assert_array_equal(self.series.lengths, np.array([3, 2, 1], dtype=np.int64))

    def test_offsets_property(self) -> None:
        np.testing.assert_array_equal(
            self.series.offsets,
            np.array([0, 3, 5, 6], dtype=np.int64),
        )

    def test_peak_indices(self) -> None:
        np.testing.assert_array_equal(
            self.series.peak_indices,
            np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        )

    def test_data_property(self) -> None:
        np.testing.assert_allclose(self.series.data, self.data)

    def test_mz_property(self) -> None:
        np.testing.assert_allclose(
            self.series.mz,
            np.array([100.0, 101.0, 102.0, 200.0, 201.0, 300.0]),
        )

    def test_intensity_property(self) -> None:
        np.testing.assert_allclose(
            self.series.intensity,
            np.array([10.0, 30.0, 20.0, 5.0, 15.0, 7.0]),
        )

    def test_metadata_property(self) -> None:
        meta = self.series.metadata
        self.assertIsNotNone(meta)
        self.assertEqual(list(meta.columns), ["annotation", "group"])
        self.assertEqual(len(meta), 6)
        self.assertEqual(meta.iloc[0]["annotation"], "a")
        self.assertEqual(meta.iloc[5]["group"], 3)

    def test_getitem_int_returns_spectrum(self) -> None:
        spectrum = self.series[0]
        self.assertIsInstance(spectrum, Spectrum)
        self.assertEqual(len(spectrum), 3)
        np.testing.assert_allclose(
            spectrum.data,
            np.array([[100.0, 10.0], [101.0, 30.0], [102.0, 20.0]], dtype=float),
        )

    def test_getitem_slice_returns_peak_series(self) -> None:
        subset = self.series[1:]
        self.assertIsInstance(subset, PeakSeries)
        self.assertEqual(len(subset), 2)
        np.testing.assert_array_equal(subset.offsets, np.array([0, 2, 3], dtype=np.int64))
        np.testing.assert_allclose(
            subset.data,
            np.array([[200.0, 5.0], [201.0, 15.0], [300.0, 7.0]], dtype=float),
        )

    def test_getitem_sequence_returns_peak_series(self) -> None:
        subset = self.series[[2, 0]]
        self.assertEqual(len(subset), 2)
        np.testing.assert_array_equal(subset.offsets, np.array([0, 1, 4], dtype=np.int64))
        np.testing.assert_allclose(
            subset.data,
            np.array(
                [
                    [300.0, 7.0],
                    [100.0, 10.0],
                    [101.0, 30.0],
                    [102.0, 20.0],
                ],
                dtype=float,
            ),
        )

    def test_n_peaks_method(self) -> None:
        self.assertEqual(self.series.n_peaks(0), 3)
        self.assertEqual(self.series.n_peaks(1), 2)
        self.assertEqual(self.series.n_peaks(2), 1)

    def test_copy_returns_independent_object(self) -> None:
        copied = self.series.copy()
        copied.intensity = np.array([1, 2, 3, 4, 5, 6], dtype=float)

        np.testing.assert_allclose(
            self.series.intensity,
            np.array([10.0, 30.0, 20.0, 5.0, 15.0, 7.0]),
        )
        np.testing.assert_allclose(
            copied.intensity,
            np.array([1, 2, 3, 4, 5, 6], dtype=float),
        )

    def test_setitem_add_metadata_column(self) -> None:
        self.series["score"] = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        self.assertIn("score", self.series.metadata_columns)
        np.testing.assert_allclose(
            self.series.metadata["score"].to_numpy(),
            np.array([0, 1, 2, 3, 4, 5], dtype=float),
        )

    def test_setitem_scalar_metadata_column(self) -> None:
        self.series["flag"] = "x"
        self.assertTrue((self.series.metadata["flag"] == "x").all())

    def test_normalize_returns_new_object(self) -> None:
        normalized = self.series.normalize(scale=1.0, in_place=False)

        expected = np.array([10.0 / 30.0, 1.0, 20.0 / 30.0, 5.0 / 15.0, 1.0, 1.0])
        np.testing.assert_allclose(normalized.intensity, expected)

        np.testing.assert_allclose(
            self.series.intensity,
            np.array([10.0, 30.0, 20.0, 5.0, 15.0, 7.0]),
        )

    def test_normalize_in_place(self) -> None:
        self.series.normalize(scale=100.0, in_place=True)
        expected = np.array(
            [
                10.0 / 30.0 * 100.0,
                100.0,
                20.0 / 30.0 * 100.0,
                5.0 / 15.0 * 100.0,
                100.0,
                100.0,
            ]
        )
        np.testing.assert_allclose(self.series.intensity, expected)

    def test_sort_by_mz(self) -> None:
        data = np.array(
            [
                [102.0, 20.0],
                [100.0, 10.0],
                [101.0, 30.0],
                [201.0, 15.0],
                [200.0, 5.0],
                [300.0, 7.0],
            ],
            dtype=float,
        )
        series = PeakSeries(
            data=data,
            offsets=self.offsets.copy(),
            metadata=self.metadata.copy(),
            metadata_columns=["annotation", "group"],
        )

        sorted_series = series.sort_by_mz()
        np.testing.assert_allclose(
            sorted_series.data,
            np.array(
                [
                    [100.0, 10.0],
                    [101.0, 30.0],
                    [102.0, 20.0],
                    [200.0, 5.0],
                    [201.0, 15.0],
                    [300.0, 7.0],
                ],
                dtype=float,
            ),
        )

    def test_sort_by_intensity(self) -> None:
        sorted_series = self.series.sort_by_intensity(ascending=False)
        np.testing.assert_allclose(
            sorted_series.data,
            np.array(
                [
                    [101.0, 30.0],
                    [102.0, 20.0],
                    [100.0, 10.0],
                    [201.0, 15.0],
                    [200.0, 5.0],
                    [300.0, 7.0],
                ],
                dtype=float,
            ),
        )

    def test_sort_by_mz_return_index(self) -> None:
        data = np.array(
            [
                [202.0, 20.0],
                [200.0, 10.0],
                [201.0, 30.0],
                [101.0, 15.0],
                [100.0, 5.0],
                [300.0, 7.0],
            ],
            dtype=float,
        )
        series = PeakSeries(
            data=data,
            offsets=self.offsets.copy(),
            metadata=self.metadata.copy(),
            metadata_columns=["annotation", "group"],
        )
        sorted_series, permutation = series.sort_by_mz(return_index=True)

        np.testing.assert_array_equal(permutation, np.array([1, 2, 0, 4, 3, 5], dtype=np.int64))
        np.testing.assert_allclose(
            sorted_series.data[:, 0],
            np.array([200.0, 201.0, 202.0, 100.0, 101.0, 300.0]),
        )

    def test_reorder(self) -> None:
        reordered = self.series.reorder([2, 0, 1])

        self.assertEqual(len(reordered), 3)
        np.testing.assert_array_equal(reordered.offsets, np.array([0, 1, 4, 6], dtype=np.int64))
        np.testing.assert_allclose(
            reordered.data,
            np.array(
                [
                    [300.0, 7.0],
                    [100.0, 10.0],
                    [101.0, 30.0],
                    [102.0, 20.0],
                    [200.0, 5.0],
                    [201.0, 15.0],
                ],
                dtype=float,
            ),
        )

    def test_iter(self) -> None:
        spectra = list(self.series)
        self.assertEqual(len(spectra), 3)
        self.assertTrue(all(isinstance(s, Spectrum) for s in spectra))


class TestSpectrum(unittest.TestCase):
    """Unit tests for Spectrum."""

    def setUp(self) -> None:
        self.data = np.array(
            [
                [100.0, 10.0],
                [101.0, 30.0],
                [102.0, 20.0],
                [200.0, 5.0],
                [201.0, 15.0],
            ],
            dtype=float,
        )
        self.offsets = np.array([0, 3, 5], dtype=np.int64)
        self.metadata = pd.DataFrame(
            {
                "annotation": ["a", "b", "c", "d", "e"],
                "group": [1, 1, 1, 2, 2],
            }
        )
        self.series = PeakSeries(
            data=self.data.copy(),
            offsets=self.offsets.copy(),
            metadata=self.metadata.copy(),
            metadata_columns=["annotation", "group"],
        )
        self.spectrum = self.series[0]

    def test_len(self) -> None:
        self.assertEqual(len(self.spectrum), 3)

    def test_data_property(self) -> None:
        np.testing.assert_allclose(
            self.spectrum.data,
            np.array([[100.0, 10.0], [101.0, 30.0], [102.0, 20.0]], dtype=float),
        )

    def test_metadata_property(self) -> None:
        metadata = self.spectrum.metadata
        self.assertIsNotNone(metadata)
        self.assertEqual(list(metadata.columns), ["annotation", "group"])
        self.assertEqual(len(metadata), 3)
        self.assertEqual(metadata.iloc[1]["annotation"], "b")

    def test_getitem_int_returns_peak(self) -> None:
        peak = self.spectrum[1]
        self.assertIsInstance(peak, Peak)
        self.assertEqual(peak.mz, 101.0)
        self.assertEqual(peak.intensity, 30.0)
        self.assertEqual(peak.metadata["annotation"], "b")

    def test_getitem_str_returns_metadata_series(self) -> None:
        series = self.spectrum["annotation"]
        self.assertIsInstance(series, pd.Series)
        self.assertEqual(series.tolist(), ["a", "b", "c"])

    def test_setitem_add_metadata_column(self) -> None:
        self.spectrum["score"] = [0.1, 0.2, 0.3]
        self.assertIn("score", self.series.metadata_columns)
        self.assertEqual(self.spectrum["score"].tolist(), [0.1, 0.2, 0.3])

    def test_setitem_scalar_metadata_column(self) -> None:
        self.spectrum["flag"] = "x"
        self.assertEqual(self.spectrum["flag"].tolist(), ["x", "x", "x"])

    def test_iter(self) -> None:
        peaks = list(self.spectrum)
        self.assertEqual(len(peaks), 3)
        self.assertTrue(all(isinstance(p, Peak) for p in peaks))

    def test_str_contains_headers(self) -> None:
        text = str(self.spectrum)
        self.assertIn("mz", text)
        self.assertIn("intensity", text)
        self.assertIn("annotation", text)

    def test_eq_true(self) -> None:
        other = self.series[0]
        self.assertEqual(self.spectrum, other)

    def test_eq_false_for_different_data(self) -> None:
        other_series = PeakSeries(
            data=np.array(
                [
                    [100.0, 10.0],
                    [101.0, 99.0],
                    [102.0, 20.0],
                ],
                dtype=float,
            ),
            offsets=np.array([0, 3], dtype=np.int64),
            metadata=pd.DataFrame(
                {"annotation": ["a", "b", "c"], "group": [1, 1, 1]}
            ),
            metadata_columns=["annotation", "group"],
        )
        other = other_series[0]
        self.assertNotEqual(self.spectrum, other)

    def test_normalize_returns_new_spectrum(self) -> None:
        normalized = self.spectrum.normalize(scale=1.0, in_place=False)
        np.testing.assert_allclose(
            normalized.intensity,
            np.array([10.0 / 30.0, 1.0, 20.0 / 30.0]),
        )
        np.testing.assert_allclose(
            self.spectrum.intensity,
            np.array([10.0, 30.0, 20.0]),
        )

    def test_normalize_in_place(self) -> None:
        self.spectrum.normalize(scale=100.0, in_place=True)
        np.testing.assert_allclose(
            self.spectrum.intensity,
            np.array([10.0 / 30.0 * 100.0, 100.0, 20.0 / 30.0 * 100.0]),
        )

    def test_sort_by_mz(self) -> None:
        data = np.array(
            [
                [102.0, 20.0],
                [100.0, 10.0],
                [101.0, 30.0],
            ],
            dtype=float,
        )
        series = PeakSeries(
            data=data,
            offsets=np.array([0, 3], dtype=np.int64),
            metadata=pd.DataFrame({"annotation": ["c", "a", "b"], "group": [1, 1, 1]}),
            metadata_columns=["annotation", "group"],
        )
        spectrum = series[0]
        sorted_spectrum = spectrum.sort_by_mz()

        np.testing.assert_allclose(
            sorted_spectrum.data,
            np.array(
                [
                    [100.0, 10.0],
                    [101.0, 30.0],
                    [102.0, 20.0],
                ],
                dtype=float,
            ),
        )
        self.assertEqual(sorted_spectrum["annotation"].tolist(), ["a", "b", "c"])

    def test_sort_by_intensity(self) -> None:
        sorted_spectrum = self.spectrum.sort_by_intensity(ascending=False)
        np.testing.assert_allclose(
            sorted_spectrum.data,
            np.array(
                [
                    [101.0, 30.0],
                    [102.0, 20.0],
                    [100.0, 10.0],
                ],
                dtype=float,
            ),
        )

    def test_mz_setter(self) -> None:
        self.spectrum.mz = np.array([110.0, 111.0, 112.0], dtype=float)
        np.testing.assert_allclose(self.spectrum.mz, np.array([110.0, 111.0, 112.0]))

    def test_intensity_setter(self) -> None:
        self.spectrum.intensity = np.array([1.0, 2.0, 3.0], dtype=float)
        np.testing.assert_allclose(self.spectrum.intensity, np.array([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()