import csv
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from msentity import MSDataset, PeakSeries, load_ms_dataset, read_csv_text, write_csv


class TestCSVIO(unittest.TestCase):
    def test_round_trip_quoting_headers_and_detection(self):
        source = MSDataset(
            pd.DataFrame({"Name": ['試料, "quoted"\nnext\tline', "empty"], "Score": [1.5, 2]}),
            PeakSeries(np.array([[10.25, 100.0], [20.5, 0.125]]), np.array([0, 2, 2])),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.CSV"
            write_csv(source, path, headers=["Score", "Name"], encoding="utf-8-sig")
            loaded = load_ms_dataset(path, spec_id_prefix="S", show_progress=False)
            self.assertEqual(loaded.columns, ["Score", "Name", "SpecID"])
            self.assertEqual(loaded.metadata.Name.tolist(), source.metadata.Name.tolist())
            np.testing.assert_array_equal(loaded[0].peaks.data, source[0].peaks.data)
            self.assertEqual(loaded[1].n_peaks, 0)
            rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
            self.assertEqual(rows[0], ["Score", "Name", "Peak"])
            self.assertEqual(rows[1][2], "10.25,100;20.5,0.125")
            self.assertEqual(len(load_ms_dataset(path, file_type="CSV")), 2)

    def test_validation(self):
        for text, error in [
            ("", "CSV text is empty"),
            ("Name\nx\n", "Peak"),
            ('Name,Name,Peak\n', "unique"),
            ('Name,Peak\nx,"100,2,3"\n', "mz,intensity"),
            ('Name,Peak\nx,"bad,2"\n', "numeric"),
            ('Name,Peak\nx,100,2\n', "Too many fields"),
        ]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, error):
                read_csv_text(text)
