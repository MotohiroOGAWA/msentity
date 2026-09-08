import io
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from msentity import MSDataset, PeakSeries
from msentity.cli.main import main
from msentity.similarity import SimilarityDataset


class TestSimilarityDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = MSDataset(
            pd.DataFrame({"SpecID": ["試料A", "試料B"]}),
            PeakSeries(np.array([[100., 4.], [200., 1.]]), np.array([0, 1, 2])),
            description="テスト", attributes={"instrument": "example"}, tags=["tag"],
        )
        self.result = SimilarityDataset.from_datasets(
            self.dataset, self.dataset[[1, 0]],
            source1="first.msds", source2="second.msds")

    def test_roundtrip_preserves_parquet_dtypes_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.mssim"
            self.result.save(path)
            with h5py.File(path) as handle:
                self.assertEqual(handle.attrs["format"], "msentity.similarity")
                parquet = pd.read_parquet(io.BytesIO(handle["table.parquet"][()].tobytes()))
                pd.testing.assert_frame_equal(parquet, self.result.table)
            loaded = SimilarityDataset.load(path)
            pd.testing.assert_frame_equal(loaded.table, self.result.table)
            self.assertEqual(loaded.metadata, self.result.metadata)
            self.assertEqual(loaded.metadata["sources"][0]["description"], "テスト")
            self.assertEqual(loaded.metadata["parameters"]["bin_width"], 0.01)
            self.assertEqual(loaded.table["index2"].tolist(), [1, 0])

    def test_failed_write_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.mssim"
            self.result.save(path)
            original = path.read_bytes()
            module = importlib.import_module("msentity.similarity.SimilarityDataset")
            with patch.object(module.os, "replace", side_effect=OSError("failure")):
                with self.assertRaises(OSError):
                    self.result.save(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_rejects_unknown_format_and_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.mssim"
            self.result.save(path)
            with h5py.File(path, "a") as handle:
                handle.attrs["schema_version"] = 999
            with self.assertRaisesRegex(ValueError, "version"):
                SimilarityDataset.load(path)
            with h5py.File(path, "a") as handle:
                handle.attrs["format"] = "other"
            with self.assertRaisesRegex(ValueError, "Not an msentity"):
                SimilarityDataset.load(path)

    def test_empty_roundtrip(self):
        result = SimilarityDataset.from_datasets(self.dataset[[]], self.dataset[[]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.mssim"
            result.save(path)
            pd.testing.assert_frame_equal(SimilarityDataset.load(path).table, result.table)

    def test_dataset_operations(self):
        filtered = self.result.filter(self.result.table["SpecID"] == "試料A")
        self.assertEqual(filtered.n_pairs, 1)
        self.assertEqual(filtered.metadata["source_row_count"], 2)
        copied = filtered.copy()
        copied.table.loc[0, "SpecID"] = "changed"
        self.assertEqual(filtered.table.loc[0, "SpecID"], "試料A")
        sorted_result = self.result.sort_values("index2")
        self.assertEqual(sorted_result.table["index2"].tolist(), [0, 1])
        self.assertIn("n_pairs=2", repr(self.result))
        statistics = self.result.describe_scores()
        self.assertEqual(statistics["count"], 2)
        self.assertEqual(statistics["q1"], 1)
        self.assertEqual(statistics["q3"], 1)
        counts, edges = self.result.histogram(2)
        np.testing.assert_array_equal(counts, [0, 2])
        np.testing.assert_allclose(edges, [0, 0.5, 1])
        with self.assertRaisesRegex(ValueError, "between 1 and 200"):
            self.result.histogram(0)

        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("parquet", "csv", "tsv"):
                path = Path(directory) / f"result.{suffix}"
                filtered.export_table(path)
                self.assertTrue(path.is_file())

    def test_cli_creates_result(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second, output = [Path(directory) / name for name in
                                     ("first.msds", "second.msds", "result.mssim")]
            self.dataset.save(first)
            self.dataset[[1, 0]].copy().save(second)
            main(["similarity-by-key", str(first), str(second), "--output", str(output),
                  "--intensity-exponent", "0.5", "--max-cum-peaks", "1"])
            result = SimilarityDataset.load(output)
            self.assertEqual(result.metadata["parameters"]["intensity_exponent"], 0.5)
            np.testing.assert_allclose(result.table["cosine_similarity"], [1, 1])
