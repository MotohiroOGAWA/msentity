"""CSV export through the viewer protocol, including legacy-package fallback."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import msentity

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "vscode-extension/python/backend.py"
sys.path.insert(0, str(BACKEND.parent))
spec = importlib.util.spec_from_file_location("csv_test_backend", BACKEND)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class CsvViewerTest(unittest.TestCase):
    def test_export_view_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            output = Path(directory) / "output.csv"
            source.write_text('Name,Score,Peak\n"a,b",2,"100,10"\nc,1,"200,20"\nd,0,\n')
            requests = [
                {"type": "export", "path": str(output), "file_type": "csv",
                 "filters": [{"column": "Name", "operator": "!=", "value": "d"}],
                 "sort": [{"column": "Score", "direction": "asc"}], "columns": ["Score", "Name"]},
                {"type": "add-dataset", "path": str(output)},
            ]
            result = subprocess.run(
                [sys.executable, str(BACKEND), str(source), "--file-type", "csv"],
                input="".join(json.dumps(r) + "\n" for r in requests), text=True,
                capture_output=True, check=True, timeout=30,
                env=dict(os.environ, PYTHONPATH=str(ROOT)),
            )
            messages = [json.loads(line.removeprefix("MSENTITY_JSON:"))
                        for line in result.stdout.splitlines() if line.startswith("MSENTITY_JSON:")]
            self.assertFalse([m for m in messages if m["type"].endswith("error")], result.stdout)
            loaded = msentity.load_ms_dataset(output)
            self.assertEqual(loaded.columns, ["Score", "Name"])
            self.assertEqual(loaded.metadata.Name.tolist(), ["c", "a,b"])

    def test_legacy_fallback(self):
        source = msentity.read_csv_text('Name,Peak\n"a,b","100,10"\nempty,\n')
        with tempfile.TemporaryDirectory() as directory:
            for file_type in ("csv", "tsv"):
                with self.subTest(file_type=file_type):
                    path = Path(directory) / f"output.{file_type}"
                    with patch.dict(msentity.__dict__):
                        delattr(msentity, f"write_{file_type}")
                        delattr(msentity, f"read_{file_type}")
                        backend.export_dataset(source, path)
                        loaded = backend.load_dataset(path)
                    self.assertEqual(loaded.metadata.Name.tolist(), ["a,b", "empty"])
                    self.assertEqual(loaded[0].peaks.mz.tolist(), [100])
                    self.assertEqual(loaded[1].n_peaks, 0)
