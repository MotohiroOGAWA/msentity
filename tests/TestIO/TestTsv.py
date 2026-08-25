import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from msentity import load_ms_dataset, read_tsv, read_tsv_text, write_tsv
from msentity.core.MSDataset import MSDataset
from msentity.core.PeakSeries import PeakSeries


class TestTSVIO(unittest.TestCase):
    def _dataset(self) -> MSDataset:
        metadata = pd.DataFrame({
            "Name": ["first", "tab\tquote \" value"],
            "Score": [1.5, 2.0],
            "Empty": [None, ""],
        })
        peaks = PeakSeries(
            np.array([[10.25, 100.0], [20.5, 0.125]], dtype=float),
            np.array([0, 2, 2], dtype=np.int64),
        )
        return MSDataset(metadata, peaks)

    def test_read_tsv_text(self) -> None:
        dataset = read_tsv_text(
            "Name\tScore\tPeak\n"
            "one\t10\t100.5,2;200,3.5\n"
            "two\tnot-numeric\t\n"
        )
        self.assertEqual(dataset.columns, ["Name", "Score"])
        self.assertEqual(len(dataset), 2)
        np.testing.assert_allclose(dataset[0].peaks.mz, [100.5, 200.0])
        np.testing.assert_allclose(dataset[0].peaks.intensity, [2.0, 3.5])
        self.assertEqual(dataset[1].n_peaks, 0)

    def test_round_trip_and_auto_detection(self) -> None:
        source = self._dataset()
        fd, path = tempfile.mkstemp(suffix=".tsv")
        os.close(fd)
        try:
            write_tsv(source, path)
            loaded = load_ms_dataset(path, show_progress=False)
            self.assertEqual(loaded.columns, source.columns)
            self.assertEqual(loaded.metadata.iloc[1]["Name"], source.metadata.iloc[1]["Name"])
            np.testing.assert_allclose(loaded[0].peaks.data, source[0].peaks.data)
            self.assertEqual(loaded[1].n_peaks, 0)
            with open(path, encoding="utf-8") as stream:
                text = stream.read()
            self.assertIn("Peak", text.splitlines()[0])
            self.assertIn("10.25,100;20.5,0.125", text)
        finally:
            os.remove(path)

    def test_requires_peak_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "Peak"):
            read_tsv_text("Name\nexample\n")

    def test_rejects_malformed_peak(self) -> None:
        with self.assertRaisesRegex(ValueError, "mz,intensity"):
            read_tsv_text("Name\tPeak\nexample\t100,2,extra\n")


if __name__ == "__main__":
    unittest.main()
