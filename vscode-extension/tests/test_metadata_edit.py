"""Metadata editing and dataset removal through the actual viewer backend."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class MetadataProtocolTest(unittest.TestCase):
    def test_edit_export_remove_and_readd(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.tsv"
            other = Path(directory) / "other.tsv"
            output = Path(directory) / "edited.msds"
            source.write_text("Name\tMass\tPeak\na\t100\t100,10\nb\t200\t200,20\n")
            other.write_text("Name\tPeak\nother\t50,5\n")
            requests = [
                {"type": "add-dataset", "path": str(other)},
                {"type": "update-metadata", "dataset_id": str(other),
                 "description": "説明", "attributes": {"測定": "test"}, "tags": ["確認済み"]},
                {"type": "page", "dataset_id": str(other)},
                {"type": "export", "dataset_id": str(other), "path": str(output), "file_type": "msds"},
                {"type": "page", "sort": [{"column": "Mass", "direction": "desc"}]},
                {"type": "update-cell", "row_id": 1, "column": "Name", "value": "changed"},
                {"type": "update-cell", "row_id": 1, "column": "Mass", "value": "201.5"},
                {"type": "update-cell", "row_id": 1, "column": "Mass", "value": "nan"},
                {"type": "update-metadata", "description": "bad", "attributes": {"a": 1}, "tags": []},
                {"type": "page"},
                {"type": "remove-dataset"},
                {"type": "remove-dataset", "dataset_id": str(other)},
                {"type": "similarity-options"},
                {"type": "add-dataset", "path": str(other)},
                {"type": "add-dataset", "path": str(output)},
            ]
            result = subprocess.run(
                [sys.executable, str(ROOT / "vscode-extension/python/backend.py"), str(source)],
                input="".join(json.dumps(r) + "\n" for r in requests),
                text=True, capture_output=True, timeout=30, check=True,
                env=dict(os.environ, PYTHONPATH=str(ROOT)))
            messages = [json.loads(line.split("MSENTITY_JSON:", 1)[1])
                        for line in result.stdout.splitlines() if line.startswith("MSENTITY_JSON:")]
            self.assertFalse([m for m in messages if m["type"] == "error"], result.stdout)
            self.assertEqual(len([m for m in messages if m["type"] == "edit-error"]), 3, result.stdout)
            pages = [m["value"] for m in messages if m["type"] == "dataset-page"]
            applied = next(m for m in messages if m["type"] == "metadata-updated")
            self.assertEqual(applied["metadata"], {
                "description": "説明", "attributes": {"測定": "test"}, "tags": ["確認済み"]})
            self.assertEqual(pages[1]["description"], "説明")
            self.assertEqual(pages[2]["row_ids"], [1, 0])
            self.assertEqual(pages[3]["rows"][0]["Name"], "a")
            self.assertEqual(pages[3]["rows"][1]["Name"], "changed")
            self.assertEqual(pages[3]["rows"][1]["Mass"], 201.5)
            self.assertEqual(pages[3]["description"], "")
            self.assertEqual(pages[4]["description"], "")
            self.assertEqual(pages[5]["description"], "説明")
            self.assertEqual(pages[5]["attributes"], {"測定": "test"})
            self.assertEqual(pages[5]["tags"], ["確認済み"])
            options = next(m for m in messages if m["type"] == "similarity-options")
            self.assertEqual(len(options["datasets"]), 1)
            self.assertTrue(other.exists())
            self.assertNotIn("changed", source.read_text())


if __name__ == "__main__":
    unittest.main()
