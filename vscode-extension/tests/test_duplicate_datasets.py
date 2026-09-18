"""Repeated imports have independent identities, edits, and display names."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class DuplicateDatasetTest(unittest.TestCase):
    def test_independent_imports_and_same_names(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.tsv"
            source.write_text("Name\tPeak\noriginal\t100,10\n")
            other = Path(directory) / "other" / "sample.tsv"
            other.parent.mkdir()
            other.write_text("Name\tPeak\nother\t200,20\n")
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "vscode-extension/python/backend.py"), str(source)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=dict(os.environ, PYTHONPATH=str(ROOT)))

            def receive(kind):
                while True:
                    line = process.stdout.readline()
                    self.assertTrue(line, f"Backend stopped before {kind}")
                    if not line.startswith("MSENTITY_JSON:"):
                        continue
                    message = json.loads(line.split("MSENTITY_JSON:", 1)[1])
                    self.assertNotIn(message["type"], ["error", "edit-error"])
                    if message["type"] == kind:
                        return message

            def send(**request):
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()

            try:
                original_id = receive("backend-ready")["dataset"]["id"]
                send(type="update-cell", dataset_id=original_id, row_id=0, column="Name", value="edited")
                receive("metadata-updated")
                send(type="add-dataset", path=str(source))
                duplicate = receive("dataset-added")["dataset"]
                self.assertNotEqual(duplicate["id"], original_id)
                self.assertEqual(duplicate["path"], str(source))
                self.assertEqual(duplicate["name"], "sample (1).tsv")
                duplicate_page = receive("dataset-page")["value"]
                self.assertEqual(duplicate_page["rows"][0]["Name"], "original")
                self.assertEqual(duplicate_page["dataset_name"], duplicate["name"])
                send(type="update-cell", dataset_id=duplicate["id"], row_id=0, column="Name", value="duplicate edited")
                receive("metadata-updated")
                send(type="page", dataset_id=original_id)
                self.assertEqual(receive("dataset-page")["value"]["rows"][0]["Name"], "edited")
                send(type="add-dataset", path=str(other))
                third = receive("dataset-added")["dataset"]
                self.assertEqual(third["name"], "sample (2).tsv")
                self.assertEqual(receive("dataset-page")["value"]["rows"][0]["Name"], "other")
                # A later addition reads the current on-disk contents again.
                source.write_text("Name\tPeak\non disk changed\t100,10\n")
                send(type="add-dataset", path=str(source))
                fourth = receive("dataset-added")["dataset"]
                self.assertEqual(fourth["name"], "sample (3).tsv")
                self.assertEqual(receive("dataset-page")["value"]["rows"][0]["Name"], "on disk changed")
                send(type="similarity-options", dataset_id=duplicate["id"])
                names = [d["name"] for d in receive("similarity-options")["datasets"]]
                self.assertEqual(len(set(names)), 4)
                send(type="reload", dataset_id=duplicate["id"])
                receive("dataset-reloaded")
                page = receive("dataset-page")["value"]
                self.assertEqual(page["dataset_name"], duplicate["name"])
                self.assertEqual(page["rows"][0]["Name"], "on disk changed")
                send(type="remove-dataset", dataset_id=original_id)
                receive("dataset-removed")
                send(type="add-dataset", path=str(source))
                replacement = receive("dataset-added")["dataset"]
                self.assertNotEqual(replacement["id"], original_id)
                self.assertEqual(replacement["name"], "sample.tsv")
                receive("dataset-page")
                send(type="page", dataset_id=duplicate["id"])
                self.assertEqual(receive("dataset-page")["value"]["dataset_name"], "sample (1).tsv")
            finally:
                process.stdin.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                process.stdout.close()
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()
