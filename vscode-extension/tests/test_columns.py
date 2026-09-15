"""Column creation, peak annotations and persistence via the viewer protocol."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ColumnsProtocolTest(unittest.TestCase):
    def test_columns_peak_edits_scope_and_msds_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.tsv"
            other = Path(directory) / "other.tsv"
            output = Path(directory) / "edited.msds"
            source.write_text("Name\tPeak\na\t200,20;100,10\nb\t50,5\n")
            other.write_text("Name\tPeak\nother\t70,7\n")
            requests = [
                {"type": "add-dataset", "path": str(other)},
                {"type": "add-column", "column": "Note"},
                {"type": "add-column", "column": "Group", "value": "sample"},
                {"type": "add-column", "column": "Note", "value": "overwrite"},
                {"type": "add-column", "column": "  "},
                {"type": "add-peak-column", "column": "Annotation"},
                {"type": "add-peak-column", "column": "Quality", "value": "unchecked"},
                {"type": "add-peak-column", "column": "mz"},
                {"type": "update-peak-cell", "row_id": 0, "peak_index": 1,
                 "column": "Annotation", "value": "fragment <A>"},
                {"type": "get-peak-record", "row_id": 0},
                {"type": "page", "filters": [{"column": "Name", "operator": "text_eq", "value": "b"}]},
                {"type": "page", "dataset_id": str(other)},
                {"type": "export", "path": str(output), "file_type": "msds"},
                {"type": "add-dataset", "path": str(output)},
                {"type": "update-peak-cell", "row_id": 0, "peak_index": 99,
                 "column": "Annotation", "value": "invalid"},
            ]
            result = subprocess.run(
                [sys.executable, str(ROOT / "vscode-extension/python/backend.py"), str(source)],
                input="".join(json.dumps(r) + "\n" for r in requests), text=True,
                capture_output=True, timeout=30, check=True,
                env=dict(os.environ, PYTHONPATH=str(ROOT)))
            messages = [json.loads(line.split("MSENTITY_JSON:", 1)[1])
                        for line in result.stdout.splitlines() if line.startswith("MSENTITY_JSON:")]
            self.assertFalse([m for m in messages if m["type"] in {"error", "export-error"}], result.stdout)
            self.assertEqual(len([m for m in messages if m["type"] == "edit-error"]), 4, result.stdout)
            records = [m for m in messages if m["type"] == "peak-record"]
            self.assertEqual(records[0]["spectrum"]["metadata"][1]["Annotation"], "fragment <A>")
            self.assertEqual(records[0]["spectrum"]["metadata"][0]["Annotation"], "")
            pages = [m["value"] for m in messages if m["type"] == "dataset-page"]
            self.assertEqual(pages[1]["rows"][0]["Note"], "")
            self.assertEqual(pages[1]["rows"][0]["Group"], "sample")
            self.assertEqual(pages[1]["spectra"][0]["metadata"][0]["Quality"], "unchecked")
            self.assertNotIn("Note", pages[2]["columns"])
            self.assertEqual(pages[2]["spectra"][0]["metadata_columns"], [])
            self.assertEqual(pages[3]["spectra"][0]["metadata"][1]["Annotation"], "fragment <A>")
            self.assertEqual(pages[3]["spectra"][1]["metadata"][0]["Annotation"], "")
            self.assertEqual(pages[3]["rows"][1]["Group"], "sample")


if __name__ == "__main__":
    unittest.main()
