"""Integration tests for calculation, persistence, filtering and histogram protocols."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from msentity.similarity import SimilarityDataset

ROOT = Path(__file__).resolve().parents[2]


def run_backend(script, source, requests):
    result = subprocess.run(
        [sys.executable, str(ROOT / "vscode-extension/python" / script), str(source), "--page-size", "1"],
        input="".join(json.dumps(request) + "\n" for request in requests),
        text=True, capture_output=True, env=dict(os.environ, PYTHONPATH=str(ROOT)),
        timeout=30, check=True,
    )
    return [json.loads(line.removeprefix("MSENTITY_JSON:"))
            for line in result.stdout.splitlines() if line.startswith("MSENTITY_JSON:")]


class SimilarityProtocolTest(unittest.TestCase):
    def test_library_search_with_loaded_and_file_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            query = Path(directory) / "query.tsv"
            reference = Path(directory) / "reference.tsv"
            loaded_output = Path(directory) / "loaded.mssim"
            file_output = Path(directory) / "file.mssim"
            query.write_text("Name\tPeak\nq-noisy\t100,1;300,10\nq-exact\t200,1\n")
            reference.write_text("Name\tPeak\nr-100\t100,1\nr-200\t200,1\nr-other\t400,1\n")
            messages = run_backend("backend.py", query, [
                {"type": "add-dataset", "path": str(reference)},
                {"type": "calculate-similarity", "mode": "library_search",
                 "dataset1": str(query), "dataset2": str(reference),
                 "path": str(loaded_output),
                 "parameters": {"threshold": 0.8, "max_pairs_per_call": 2}},
                {"type": "calculate-similarity", "mode": "library_search",
                 "dataset1": str(query), "reference_path": str(reference),
                 "path": str(file_output),
                 "parameters": {"threshold": 0.8, "method": "reverse_cosine",
                                "include_matched_data": False, "max_pairs_per_call": 2}},
            ])
            self.assertEqual(sum(m["type"] == "similarity-complete" for m in messages), 2)
            progress = [m for m in messages if m["type"] == "similarity-progress"]
            self.assertEqual(progress[-1]["percent"], 100)
            loaded = SimilarityDataset.load(loaded_output)
            self.assertTrue(loaded.has_matched_data)
            self.assertEqual(loaded.table[["index1", "index2"]].values.tolist(), [[1, 1]])
            viewer_messages = run_backend("similarity_backend.py", loaded_output, [
                {"type": "page"},
                {"type": "match", "row": 0},
            ])
            page = next(message["value"] for message in viewer_messages
                        if message["type"] == "similarity-page")
            self.assertTrue(page["has_matched_data"])
            self.assertEqual(page["matched_data_rows"], [1, 1])
            match = next(message for message in viewer_messages
                         if message["type"] == "similarity-match")
            self.assertEqual(match["query"]["row"]["Name"], "q-exact")
            self.assertEqual(match["reference"]["row"]["Name"], "r-200")
            file_result = SimilarityDataset.load(file_output)
            self.assertFalse(file_result.has_matched_data)
            self.assertEqual(file_result.table[["index1", "index2"]].values.tolist(),
                             [[0, 0], [1, 1]])

    def test_calculate_select_datasets_reload_filter_histogram_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second, third, output, subset, tsv, parquet = [Path(directory) / name for name in
                ("first.tsv", "second.tsv", "third.tsv", "result.mssim", "subset.mssim", "subset.tsv", "subset.parquet")]
            first.write_text("SpecID\tPeak\na\t100,4;101,1\nb\t200,1\n")
            second.write_text("OtherID\tPeak\nb\t300,1\na\t100,1\n")
            third.write_text("SpecID\tPeak\nduplicate\t100,1\nduplicate\t101,1\n")
            messages = run_backend("backend.py", first, [
                {"type": "add-dataset", "path": str(second)},
                {"type": "add-dataset", "path": str(third)},
                {"type": "similarity-options"},
                {"type": "calculate-similarity", "dataset1": str(first), "dataset2": str(second),
                 "path": str(output), "parameters": {"key2": "OtherID", "intensity_exponent": 0.5, "max_cum_peaks": 1}},
                {"type": "calculate-similarity", "dataset1": str(first), "dataset2": str(third),
                 "path": str(output)},
                {"type": "calculate-similarity", "dataset1": str(first), "dataset2": str(first),
                 "path": str(output)},
            ])
            self.assertFalse([m for m in messages if m["type"] == "error"], messages)
            choices = next(m["datasets"] for m in messages if m["type"] == "similarity-options")
            self.assertEqual(len(choices), 3)
            self.assertIn("OtherID", choices[1]["columns"])
            self.assertEqual(sum(m["type"] == "similarity-complete" for m in messages), 1)
            failures = [m for m in messages if m["type"] == "similarity-error"]
            self.assertEqual(len(failures), 2)
            self.assertIn("one-to-one", failures[0]["message"])
            result = SimilarityDataset.load(output)
            self.assertAlmostEqual(result.table.cosine_similarity[0], 2 / (5 ** .5), places=6)
            self.assertEqual(result.table.index2.tolist(), [1, 0])

            filters = [{"column": "cosine_similarity", "operator": ">=", "value": "0.8"}]
            messages = run_backend("similarity_backend.py", output, [
                {"type": "page", "bins": 2},
                {"type": "page", "page": 1, "bins": 2},
                {"type": "page", "filters": filters, "bins": 2},
                *[{"type": "export", "filters": filters, "path": str(path)} for path in (subset, tsv, parquet)],
                {"type": "reload", "sort": {"column": "cosine_similarity", "direction": "asc"}},
                {"type": "page", "filters": [{"column": "SpecID", "operator": "text_eq", "value": "none"}]},
                {"type": "page", "bins": 0},
                {"type": "page", "filters": [{"column": "cosine_similarity", "operator": ">", "value": "bad"}]},
                {"type": "page"},
            ])
            pages = [m["value"] for m in messages if m["type"] == "similarity-page"]
            self.assertEqual(pages[0]["histogram"]["counts"], [1, 1])  # whole table, not one-row page
            self.assertAlmostEqual(pages[0]["statistics"]["q1"], 0.2236068, places=6)
            self.assertAlmostEqual(pages[0]["statistics"]["q3"], 0.6708204, places=6)
            self.assertEqual(pages[1]["rows"][0]["SpecID"], "b")
            self.assertEqual(pages[2]["histogram"]["counts"], [0, 1])
            self.assertEqual(pages[3]["rows"][0]["SpecID"], "b")
            self.assertEqual(pages[4]["total_rows"], 0)
            self.assertIsNone(pages[4]["statistics"]["mean"])
            self.assertEqual(pages[5]["total_rows"], 2)  # recover after request errors
            self.assertEqual(sum(m["type"] == "error" for m in messages), 2)
            loaded = SimilarityDataset.load(subset)
            self.assertEqual(loaded.table.SpecID.tolist(), ["a"])
            self.assertEqual(loaded.metadata["export_view"]["filters"], filters)
            self.assertEqual(pd.read_csv(tsv, sep="\t").SpecID.tolist(), ["a"])
            self.assertEqual(pd.read_parquet(parquet).SpecID.tolist(), ["a"])

    def test_histogram_includes_one_and_does_not_read_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "standalone.mssim"
            SimilarityDataset(pd.DataFrame({"index1": [0, 1], "index2": [0, 1],
                                           "cosine_similarity": [0., 1.]}),
                             {"sources": [{"path": "/nonexistent/source.msds"}]}).save(output)
            messages = run_backend("similarity_backend.py", output, [{"type": "page", "bins": 2}])
            self.assertEqual(messages[1]["value"]["histogram"]["counts"], [1, 1])


if __name__ == "__main__":
    unittest.main()
