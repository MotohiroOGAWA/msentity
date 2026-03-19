import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from msentity.core.MSDataset import MSDataset, MSDatasetMeta, SpectrumRecord
from msentity.core.PeakSeries import PeakSeries, Spectrum


class TestMSDataset(unittest.TestCase):
    """Unit tests for MSDataset."""

    def setUp(self) -> None:
        self.peak_data = np.array(
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

        self.peak_metadata = pd.DataFrame(
            {
                "annotation": ["a", "b", "c", "d", "e", "f"],
                "peak_group": [1, 1, 1, 2, 2, 3],
            }
        )

        self.spectrum_metadata = pd.DataFrame(
            {
                "name": ["spec1", "spec2", "spec3"],
                "precursor_mz": [150.0, 250.0, 350.0],
                "SMILES": ["CCO", "CCC", "CCN"],
            }
        )

        self.peak_series = PeakSeries(
            data=self.peak_data.copy(),
            offsets=self.offsets.copy(),
            metadata=self.peak_metadata.copy(),
            metadata_columns=["annotation", "peak_group"],
        )

        self.dataset = MSDataset(
            spectrum_metadata=self.spectrum_metadata.copy(),
            peak_series=self.peak_series,
            columns=["name", "precursor_mz", "SMILES"],
            description="example dataset",
            attributes={"source": "unit-test"},
            tags=["test", "demo"],
        )

    def test_len(self) -> None:
        self.assertEqual(len(self.dataset), 3)
        self.assertEqual(self.dataset.n_rows, 3)

    def test_shape(self) -> None:
        self.assertEqual(self.dataset.shape, (3, 3))
        self.assertEqual(self.dataset.n_columns, 3)

    def test_n_peaks_total(self) -> None:
        self.assertEqual(self.dataset.n_peaks_total, 6)

    def test_columns_property(self) -> None:
        self.assertEqual(self.dataset.columns, ["name", "precursor_mz", "SMILES"])

    def test_description_property(self) -> None:
        self.assertEqual(self.dataset.description, "example dataset")

    def test_attributes_property(self) -> None:
        self.assertEqual(self.dataset.attributes, {"source": "unit-test"})

    def test_tags_property(self) -> None:
        self.assertEqual(self.dataset.tags, ["test", "demo"])

    def test_set_attribute(self) -> None:
        self.dataset.set_attribute("instrument", "orbitrap")
        self.assertTrue(self.dataset.has_attribute("instrument"))
        self.assertEqual(self.dataset.attributes["instrument"], "orbitrap")

    def test_remove_attribute(self) -> None:
        self.assertTrue(self.dataset.remove_attribute("source"))
        self.assertFalse(self.dataset.has_attribute("source"))

    def test_clear_attributes(self) -> None:
        self.dataset.clear_attributes()
        self.assertEqual(self.dataset.attributes, {})

    def test_add_tag(self) -> None:
        self.assertTrue(self.dataset.add_tag("new"))
        self.assertIn("new", self.dataset.tags)

    def test_add_duplicate_tag(self) -> None:
        self.assertFalse(self.dataset.add_tag("test"))

    def test_remove_tag(self) -> None:
        self.assertTrue(self.dataset.remove_tag("demo"))
        self.assertNotIn("demo", self.dataset.tags)

    def test_clear_tags(self) -> None:
        self.dataset.clear_tags()
        self.assertEqual(self.dataset.tags, [])

    def test_metadata_property(self) -> None:
        metadata = self.dataset.metadata
        self.assertEqual(list(metadata.columns), ["name", "precursor_mz", "SMILES"])
        self.assertEqual(len(metadata), 3)
        self.assertEqual(metadata.iloc[0]["name"], "spec1")

    def test_peaks_property(self) -> None:
        self.assertIsInstance(self.dataset.peaks, PeakSeries)
        self.assertEqual(self.dataset.peaks.n_peaks_total, 6)

    def test_getitem_int_returns_spectrum_record(self) -> None:
        record = self.dataset[0]
        self.assertIsInstance(record, SpectrumRecord)
        self.assertEqual(record["name"], "spec1")

    def test_getitem_str_returns_series(self) -> None:
        series = self.dataset["name"]
        self.assertIsInstance(series, pd.Series)
        self.assertEqual(series.tolist(), ["spec1", "spec2", "spec3"])

    def test_getitem_slice_returns_dataset(self) -> None:
        subset = self.dataset[1:]
        self.assertIsInstance(subset, MSDataset)
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset["name"].tolist(), ["spec2", "spec3"])
        np.testing.assert_array_equal(subset.peaks.offsets, np.array([0, 2, 3], dtype=np.int64))

    def test_getitem_sequence_returns_dataset(self) -> None:
        subset = self.dataset[[2, 0]]
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset["name"].tolist(), ["spec3", "spec1"])

    def test_getitem_boolean_series_returns_dataset(self) -> None:
        mask = pd.Series([True, False, True])
        subset = self.dataset[mask]
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset["name"].tolist(), ["spec1", "spec3"])

    def test_setitem_add_metadata_column(self) -> None:
        self.dataset["score"] = [0.1, 0.2, 0.3]
        self.assertIn("score", self.dataset.columns)
        np.testing.assert_allclose(
            self.dataset["score"].to_numpy(dtype=float),
            np.array([0.1, 0.2, 0.3])
        )

    def test_setitem_scalar_metadata_column(self) -> None:
        self.dataset["flag"] = "ok"
        self.assertEqual(self.dataset["flag"].tolist(), ["ok", "ok", "ok"])

    def test_copy_returns_independent_dataset(self) -> None:
        copied = self.dataset.copy()
        copied["name"] = ["x", "y", "z"]

        self.assertEqual(self.dataset["name"].tolist(), ["spec1", "spec2", "spec3"])
        self.assertEqual(copied["name"].tolist(), ["x", "y", "z"])

    def test_sort_by(self) -> None:
        sorted_ds = self.dataset.sort_by("precursor_mz", ascending=False)
        self.assertEqual(sorted_ds["name"].tolist(), ["spec3", "spec2", "spec1"])

    def test_concat(self) -> None:
        ds1 = self.dataset[:2].copy()
        ds2 = self.dataset[2:].copy()
        merged = MSDataset.concat([ds1, ds2], description="merged")

        self.assertEqual(len(merged), 3)
        self.assertEqual(merged.description, "merged")
        self.assertEqual(merged["name"].tolist(), ["spec1", "spec2", "spec3"])
        self.assertEqual(merged.n_peaks_total, 6)

    def test_merge_metadata(self) -> None:
        right = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCC", "CCN"],
                "class": ["alcohol", "alkane", "amine"],
            }
        )
        self.dataset.merge_metadata(right, on="SMILES")
        self.assertIn("class", self.dataset.columns)
        self.assertEqual(self.dataset["class"].tolist(), ["alcohol", "alkane", "amine"])

    def test_merge_metadata_with_prefix(self) -> None:
        right = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCC", "CCN"],
                "class": ["alcohol", "alkane", "amine"],
            }
        )
        self.dataset.merge_metadata(right, on="SMILES", right_prefix="cf_")
        self.assertIn("cf_class", self.dataset.columns)
        self.assertEqual(self.dataset["cf_class"].tolist(), ["alcohol", "alkane", "amine"])

    def test_to_hdf5_and_from_hdf5(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dataset.h5")

            self.dataset.to_hdf5(path)
            loaded = MSDataset.from_hdf5(path)

            self.assertEqual(len(loaded), len(self.dataset))
            self.assertEqual(loaded.description, self.dataset.description)
            self.assertEqual(loaded.attributes, self.dataset.attributes)
            self.assertEqual(loaded.tags, self.dataset.tags)
            self.assertEqual(loaded["name"].tolist(), self.dataset["name"].tolist())
            np.testing.assert_allclose(loaded.peaks.data, self.dataset.peaks.data)

    def test_read_dataset_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dataset.h5")

            self.dataset.to_hdf5(path)
            meta = MSDataset.read_dataset_meta(path)

            self.assertIsInstance(meta, MSDatasetMeta)
            self.assertEqual(meta.description, "example dataset")
            self.assertEqual(meta.attributes, {"source": "unit-test"})
            self.assertEqual(meta.tags, ["test", "demo"])


class TestSpectrumRecord(unittest.TestCase):
    """Unit tests for SpectrumRecord."""

    def setUp(self) -> None:
        peak_data = np.array(
            [
                [100.0, 10.0],
                [101.0, 30.0],
                [102.0, 20.0],
                [200.0, 5.0],
                [201.0, 15.0],
            ],
            dtype=float,
        )
        offsets = np.array([0, 3, 5], dtype=np.int64)

        peak_metadata = pd.DataFrame(
            {
                "annotation": ["a", "b", "c", "d", "e"],
                "peak_group": [1, 1, 1, 2, 2],
            }
        )
        spectrum_metadata = pd.DataFrame(
            {
                "name": ["spec1", "spec2"],
                "precursor_mz": [150.0, 250.0],
                "SMILES": ["CCO", "CCC"],
            }
        )

        peak_series = PeakSeries(
            data=peak_data,
            offsets=offsets,
            metadata=peak_metadata,
            metadata_columns=["annotation", "peak_group"],
        )

        self.dataset = MSDataset(
            spectrum_metadata=spectrum_metadata,
            peak_series=peak_series,
            columns=["name", "precursor_mz", "SMILES"],
        )
        self.record = self.dataset[0]

    def test_repr(self) -> None:
        text = repr(self.record)
        self.assertIn("SpectrumRecord", text)
        self.assertIn("n_peaks=3", text)

    def test_str(self) -> None:
        text = str(self.record)
        self.assertIn("name", text)
        self.assertIn("spec1", text)
        self.assertIn("mz", text)

    def test_contains(self) -> None:
        self.assertIn("name", self.record)
        self.assertNotIn("unknown", self.record)

    def test_getitem(self) -> None:
        self.assertEqual(self.record["name"], "spec1")
        self.assertEqual(self.record["precursor_mz"], 150.0)

    def test_setitem_existing_column(self) -> None:
        self.record["name"] = "updated"
        self.assertEqual(self.record["name"], "updated")
        self.assertEqual(self.dataset["name"].tolist(), ["updated", "spec2"])

    def test_setitem_new_column(self) -> None:
        self.record["score"] = 0.95
        self.assertIn("score", self.dataset.columns)
        self.assertEqual(self.record["score"], 0.95)
        self.assertTrue(pd.isna(self.dataset[1]["score"]))

    def test_eq_true(self) -> None:
        other = self.dataset[0]
        self.assertEqual(self.record, other)

    def test_eq_false(self) -> None:
        other = self.dataset[1]
        self.assertNotEqual(self.record, other)

    def test_columns_property(self) -> None:
        self.assertEqual(self.record.columns, ["name", "precursor_mz", "SMILES"])

    def test_n_peaks(self) -> None:
        self.assertEqual(self.record.n_peaks, 3)

    def test_spectrum_property(self) -> None:
        self.assertIsInstance(self.record.spectrum, Spectrum)
        np.testing.assert_allclose(
            self.record.spectrum.data,
            np.array([[100.0, 10.0], [101.0, 30.0], [102.0, 20.0]], dtype=float),
        )

    def test_is_integer_mz_false(self) -> None:
        self.assertTrue(self.record.is_integer_mz)

    def test_is_integer_mz_true(self) -> None:
        self.record.spectrum.mz = np.array([100.0, 101.0, 102.0], dtype=float)
        self.assertTrue(self.record.is_integer_mz)

    def test_normalize_returns_new_record(self) -> None:
        normalized = self.record.normalize(scale=1.0, in_place=False)
        np.testing.assert_allclose(
            normalized.spectrum.intensity,
            np.array([10.0 / 30.0, 1.0, 20.0 / 30.0]),
        )
        np.testing.assert_allclose(
            self.record.spectrum.intensity,
            np.array([10.0, 30.0, 20.0]),
        )

    def test_normalize_in_place(self) -> None:
        self.record.normalize(scale=100.0, in_place=True)
        np.testing.assert_allclose(
            self.record.spectrum.intensity,
            np.array([10.0 / 30.0 * 100.0, 100.0, 20.0 / 30.0 * 100.0]),
        )

    def test_sort_by_mz(self) -> None:
        self.record.spectrum.data = np.array(
            [
                [102.0, 20.0],
                [100.0, 10.0],
                [101.0, 30.0],
            ],
            dtype=float,
        )
        sorted_record = self.record.sort_by_mz(in_place=False)

        np.testing.assert_allclose(
            sorted_record.spectrum.mz,
            np.array([100.0, 101.0, 102.0], dtype=float),
        )

    def test_sort_by_intensity(self) -> None:
        sorted_record = self.record.sort_by_intensity(ascending=False, in_place=False)
        intensity = sorted_record.spectrum.intensity

        self.assertTrue(np.all(intensity[:-1] >= intensity[1:]))

    def test_sort_by_mz_in_place(self) -> None:
        self.record.spectrum.data = np.array(
            [
                [102.0, 20.0],
                [100.0, 10.0],
                [101.0, 30.0],
            ],
            dtype=float,
        )
        self.record.sort_by_mz(in_place=True)
        np.testing.assert_allclose(
            self.record.spectrum.mz,
            np.array([100.0, 101.0, 102.0], dtype=float),
        )

    def test_copy(self) -> None:
        copied = self.record.copy()
        self.assertIsInstance(copied, SpectrumRecord)
        self.assertEqual(copied["name"], self.record["name"])
        self.assertEqual(copied.n_peaks, self.record.n_peaks)


if __name__ == "__main__":
    unittest.main()