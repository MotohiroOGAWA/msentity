import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import h5py

from msentity.processing.inspect import print_hdf5_structure


class TestInspectFunctions(unittest.TestCase):
    """Unit tests for inspection utilities."""

    def _create_test_hdf5(self) -> str:
        """Create a temporary HDF5 file for testing."""
        fd, path = tempfile.mkstemp(suffix=".h5")
        os.close(fd)

        with h5py.File(path, "w") as f:
            f.attrs["version"] = "1.0"

            grp_a = f.create_group("group_a")
            grp_a.attrs["description"] = "first group"

            grp_b = grp_a.create_group("group_b")

            ds1 = grp_a.create_dataset("dataset1", data=[1, 2, 3])
            ds1.attrs["unit"] = "a.u."

            ds2 = grp_b.create_dataset("dataset2", data=[[1.0, 2.0], [3.0, 4.0]])
            ds2.attrs["kind"] = "matrix"

        return path

    def test_print_hdf5_structure_basic(self) -> None:
        """print_hdf5_structure should print groups and datasets."""
        path = self._create_test_hdf5()
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_hdf5_structure(path)

            out = buf.getvalue()

            self.assertIn("HDF5 file:", out)
            self.assertIn("[Group] /", out)
            self.assertIn("[Group] /group_a", out)
            self.assertIn("[Group] /group_a/group_b", out)
            self.assertIn("[Dataset] /group_a/dataset1", out)
            self.assertIn("[Dataset] /group_a/group_b/dataset2", out)
        finally:
            os.remove(path)

    def test_print_hdf5_structure_with_attrs(self) -> None:
        """print_hdf5_structure should print attributes when show_attrs=True."""
        path = self._create_test_hdf5()
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_hdf5_structure(path, show_attrs=True)

            out = buf.getvalue()

            self.assertIn("@attr version:", out)
            self.assertIn("@attr description:", out)
            self.assertIn("@attr unit:", out)
            self.assertIn("@attr kind:", out)
        finally:
            os.remove(path)

    def test_print_hdf5_structure_without_attrs(self) -> None:
        """print_hdf5_structure should omit attributes when show_attrs=False."""
        path = self._create_test_hdf5()
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_hdf5_structure(path, show_attrs=False)

            out = buf.getvalue()

            self.assertNotIn("@attr", out)
        finally:
            os.remove(path)

    def test_print_hdf5_structure_without_dataset_details(self) -> None:
        """print_hdf5_structure should omit shape and dtype when show_datasets=False."""
        path = self._create_test_hdf5()
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_hdf5_structure(path, show_datasets=False)

            out = buf.getvalue()

            self.assertIn("[Dataset] /group_a/dataset1", out)
            self.assertIn("[Dataset] /group_a/group_b/dataset2", out)
            self.assertNotIn("shape=", out)
            self.assertNotIn("dtype=", out)
        finally:
            os.remove(path)

    def test_print_hdf5_structure_max_depth_zero(self) -> None:
        """max_depth=0 should print only the root group."""
        path = self._create_test_hdf5()
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_hdf5_structure(path, max_depth=0)

            out = buf.getvalue()

            self.assertIn("[Group] /", out)
            self.assertNotIn("/group_a", out)
            self.assertNotIn("dataset1", out)
            self.assertNotIn("dataset2", out)
        finally:
            os.remove(path)

    def test_print_hdf5_structure_max_depth_one(self) -> None:
        """max_depth=1 should print direct children of the root group only."""
        path = self._create_test_hdf5()
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_hdf5_structure(path, max_depth=1)

            out = buf.getvalue()

            self.assertIn("[Group] /", out)
            self.assertIn("[Group] /group_a", out)
            self.assertNotIn("/group_a/group_b", out)
            self.assertNotIn("dataset1", out)
            self.assertNotIn("dataset2", out)
        finally:
            os.remove(path)

    def test_print_hdf5_structure_invalid_max_depth(self) -> None:
        """print_hdf5_structure should raise ValueError for negative max_depth."""
        path = self._create_test_hdf5()
        try:
            with self.assertRaises(ValueError):
                print_hdf5_structure(path, max_depth=-1)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()