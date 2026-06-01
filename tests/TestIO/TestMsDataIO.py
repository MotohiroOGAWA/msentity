import unittest

from msentity import load_ms_data

from tests.common import (
    SAMPLE_HDF5_FILE,
    SAMPLE_MGF_FILE,
    SAMPLE_MSP_FILE,
)


class TestLoadMSData(unittest.TestCase):
    """Unit tests for load_ms_data."""

    def test_load_msp_by_extension(self) -> None:
        """load_ms_data should load an MSP file when extension is .msp."""
        dataset = load_ms_data(str(SAMPLE_MSP_FILE))

        self.assertGreater(len(dataset), 0)

    def test_load_mgf_by_extension(self) -> None:
        """load_ms_data should load an MGF file when extension is .mgf."""
        dataset = load_ms_data(str(SAMPLE_MGF_FILE))

        self.assertGreater(len(dataset), 0)

    def test_load_hdf5_by_h5_extension(self) -> None:
        """load_ms_data should load an HDF5 file when extension is .h5."""
        dataset = load_ms_data(str(SAMPLE_HDF5_FILE))

        self.assertGreater(len(dataset), 0)

    def test_load_with_explicit_file_type(self) -> None:
        """load_ms_data should use file_type even if extension is different."""
        dataset = load_ms_data(
            str(SAMPLE_MSP_FILE),
            file_type="msp",
        )

        self.assertGreater(len(dataset), 0)

    def test_load_msp_with_spec_id_prefix(self) -> None:
        """load_ms_data should assign SpecID with the given prefix."""
        dataset = load_ms_data(
            str(SAMPLE_MSP_FILE),
            spec_id_prefix="test",
        )

        self.assertIn("SpecID", dataset.columns)

        first_spec_id = dataset[0]["SpecID"]
        self.assertTrue(str(first_spec_id).startswith("test"))

    def test_unsupported_file_type_raises_value_error(self) -> None:
        """load_ms_data should raise ValueError for unsupported file_type."""
        with self.assertRaises(ValueError) as context:
            load_ms_data(
                str(SAMPLE_MSP_FILE),
                file_type="csv",
            )

        self.assertIn("Unsupported file type", str(context.exception))


if __name__ == "__main__":
    unittest.main()