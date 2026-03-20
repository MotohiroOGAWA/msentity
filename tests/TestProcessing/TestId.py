import unittest

import numpy as np
import pandas as pd

from msentity.core.MSDataset import MSDataset
from msentity.core.PeakSeries import PeakSeries
from msentity.processing.id import set_peak_id, set_spec_id


class TestIDFunctions(unittest.TestCase):
    """Unit tests for ID assignment utilities."""

    def setUp(self) -> None:
        self.dataset = self._make_dataset()

    def _make_dataset(self) -> MSDataset:
        """Create a small MSDataset for testing."""
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
        return self.dataset

    def test_set_spec_id_default(self) -> None:
        """set_spec_id should create the default SpecID column."""
        result = set_spec_id(self.dataset)
        self.assertTrue(result)

        self.assertIn("SpecID", self.dataset._spectrum_metadata_ref.columns)
        self.assertEqual(
            self.dataset._spectrum_metadata_ref["SpecID"].tolist(),
            ["1", "2", "3"] if len(str(len(self.dataset))) == 1 else ["01", "02", "03"],
        )

    def test_set_spec_id_with_prefix_and_custom_column(self) -> None:
        """set_spec_id should support custom column names and prefixes."""
        result = set_spec_id(
            self.dataset,
            col_name="SpectrumID",
            prefix="SP",
            start=1,
        )
        self.assertTrue(result)

        self.assertIn("SpectrumID", self.dataset._spectrum_metadata_ref.columns)
        self.assertEqual(
            self.dataset._spectrum_metadata_ref["SpectrumID"].tolist(),
            ["SP1", "SP2", "SP3"] if len(str(3)) == 1 else ["SP01", "SP02", "SP03"],
        )

    def test_set_spec_id_skip_when_column_exists(self) -> None:
        """set_spec_id should return False when the target column already exists."""
        result1 = set_spec_id(self.dataset, col_name="SpecID")
        result2 = set_spec_id(self.dataset, col_name="SpecID", overwrite=False)

        self.assertTrue(result1)
        self.assertFalse(result2)

    def test_set_spec_id_overwrite(self) -> None:
        """set_spec_id should overwrite an existing column when overwrite=True."""
        self.dataset._spectrum_metadata_ref["SpecID"] = ["X", "Y", "Z"]

        result = set_spec_id(
            self.dataset,
            col_name="SpecID",
            prefix="S",
            overwrite=True,
            start=1,
        )
        self.assertTrue(result)

        self.assertEqual(
            self.dataset._spectrum_metadata_ref["SpecID"].tolist(),
            ["S1", "S2", "S3"] if len(str(3)) == 1 else ["S01", "S02", "S03"],
        )

    def test_set_spec_id_invalid_prefix(self) -> None:
        """set_spec_id should raise TypeError when prefix is not a string."""
        with self.assertRaises(TypeError):
            set_spec_id(self.dataset, prefix=123)  # type: ignore[arg-type]

    def test_set_spec_id_invalid_col_name(self) -> None:
        """set_spec_id should raise TypeError when col_name is not a string."""
        with self.assertRaises(TypeError):
            set_spec_id(self.dataset, col_name=123)  # type: ignore[arg-type]

    def test_set_spec_id_invalid_start(self) -> None:
        """set_spec_id should raise ValueError when start is invalid."""
        with self.assertRaises(ValueError):
            set_spec_id(self.dataset, start=-1)

    def test_set_peak_id_default(self) -> None:
        """set_peak_id should assign local peak IDs that reset for each spectrum."""
        result = set_peak_id(self.dataset)
        self.assertTrue(result)

        self.assertIn("PeakID", self.dataset.peaks._metadata_ref.columns)
        self.assertEqual(
            self.dataset.peaks._metadata_ref["PeakID"].tolist(),
            ["0", "1", "2", "0", "1", "0"],
        )

    def test_set_peak_id_custom_column_and_start(self) -> None:
        """set_peak_id should support custom column names and starting values."""
        result = set_peak_id(
            self.dataset,
            col_name="LocalPeakID",
            start=1,
        )
        self.assertTrue(result)

        self.assertIn("LocalPeakID", self.dataset.peaks._metadata_ref.columns)
        self.assertEqual(
            self.dataset.peaks._metadata_ref["LocalPeakID"].tolist(),
            ["1", "2", "3", "1", "2", "1"],
        )

    def test_set_peak_id_skip_when_column_exists(self) -> None:
        """set_peak_id should return False when the target column already exists."""
        result1 = set_peak_id(self.dataset, col_name="PeakID")
        result2 = set_peak_id(self.dataset, col_name="PeakID", overwrite=False)

        self.assertTrue(result1)
        self.assertFalse(result2)

    def test_set_peak_id_overwrite(self) -> None:
        """set_peak_id should overwrite an existing column when overwrite=True."""
        self.dataset.peaks._metadata_ref["PeakID"] = ["x"] * len(self.dataset.peaks._metadata_ref)

        result = set_peak_id(
            self.dataset,
            col_name="PeakID",
            overwrite=True,
            start=1,
        )
        self.assertTrue(result)

        self.assertEqual(
            self.dataset.peaks._metadata_ref["PeakID"].tolist(),
            ["1", "2", "3", "1", "2", "1"],
        )


if __name__ == "__main__":
    unittest.main()