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
from msentity.similarity import SimilarityDataset, calculate_library_search


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
                self.assertEqual(handle.attrs["schema_version"], 2)
                self.assertEqual(handle.attrs["data_mode"], "lightweight")
                parquet = pd.read_parquet(io.BytesIO(handle["table.parquet"][()].tobytes()))
                pd.testing.assert_frame_equal(parquet, self.result.table)
            loaded = SimilarityDataset.load(path)
            pd.testing.assert_frame_equal(loaded.table, self.result.table)
            self.assertEqual(loaded.metadata, self.result.metadata)
            self.assertEqual(loaded.metadata["sources"][0]["description"], "テスト")
            self.assertEqual(loaded.metadata["parameters"]["bin_width"], 0.01)
            self.assertEqual(loaded.table["index2"].tolist(), [1, 0])

    def test_loads_schema_version_one_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.mssim"
            self.result.save(path)
            with h5py.File(path, "a") as handle:
                handle.attrs["schema_version"] = 1
                del handle.attrs["data_mode"]
            loaded = SimilarityDataset.load(path)
            pd.testing.assert_frame_equal(loaded.table, self.result.table)
            self.assertFalse(loaded.has_matched_data)

    def test_library_result_embeds_unique_matches_and_roundtrips(self):
        result = calculate_library_search(
            self.dataset,
            self.dataset,
            threshold=0,
            max_pairs_per_call=1,
        )
        self.assertEqual(len(result), 4)
        self.assertTrue(result.has_matched_data)
        self.assertEqual([len(dataset) for dataset in result.matched_datasets], [2, 2])
        self.assertEqual(result.table["data_index1"].tolist(), [0, 0, 1, 1])
        self.assertEqual(result.table["data_index2"].tolist(), [0, 1, 0, 1])
        query_record, reference_record = result.match_records(1)
        self.assertEqual(query_record["SpecID"], "試料A")
        self.assertEqual(reference_record["SpecID"], "試料B")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.mssim"
            result.save(path)
            with h5py.File(path) as handle:
                self.assertEqual(handle.attrs["data_mode"], "embedded")
                self.assertEqual(sorted(handle["matched_data"].keys()), ["1", "2"])
            loaded = SimilarityDataset.load(path)
            pd.testing.assert_frame_equal(loaded.table, result.table)
            self.assertEqual(loaded.matched_datasets[0].metadata["SpecID"].tolist(),
                             ["試料A", "試料B"])
            np.testing.assert_array_equal(loaded.matched_source_indices[1], [0, 1])

    def test_filtered_embedded_result_saves_only_referenced_records(self):
        result = calculate_library_search(self.dataset, self.dataset, threshold=0)
        filtered = result.filter(result.table.index == 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filtered.mssim"
            filtered.save(path)
            loaded = SimilarityDataset.load(path)
        self.assertEqual([len(dataset) for dataset in loaded.matched_datasets], [1, 1])
        self.assertEqual(loaded.table[["data_index1", "data_index2"]].values.tolist(), [[0, 0]])

    def test_library_cli_defaults_to_embedded_and_supports_lightweight(self):
        with tempfile.TemporaryDirectory() as directory:
            query = Path(directory) / "query.msds"
            reference = Path(directory) / "reference.msds"
            embedded = Path(directory) / "embedded.mssim"
            lightweight = Path(directory) / "lightweight.mssim"
            self.dataset.save(query)
            self.dataset.save(reference)
            main(["library-search", str(query), str(reference), "--threshold", "1",
                  "--output", str(embedded)])
            main(["library-search", str(query), str(reference), "--threshold", "1",
                  "--lightweight", "--output", str(lightweight)])
            self.assertTrue(SimilarityDataset.load(embedded).has_matched_data)
            self.assertFalse(SimilarityDataset.load(lightweight).has_matched_data)

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
