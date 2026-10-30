"""Tests of the Python binding. Run from bindings/python:

    python -m unittest discover -s tests -v

The C library is taken from MSENTITY_C_LIBRARY or ../../build.
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import msentity as ms

# Make sure the binding from this repository is the one being tested (or,
# with MSENTITY_TEST_INSTALLED=1, an installed wheel of it).
if not os.environ.get("MSENTITY_TEST_INSTALLED"):
    assert Path(ms.__file__).resolve().parent == Path(__file__).resolve().parents[1] / "msentity", (
        f"imported {ms.__file__}, not the binding; run from bindings/python or set PYTHONPATH"
    )
assert hasattr(ms, "Dataset"), f"{ms.__file__} is not the msentity binding"

REPO = Path(__file__).resolve().parents[3]
EXAMPLE = REPO / "docs" / "examples" / "example.msp"
DATA = REPO / "tests" / "data"


class TestLoading(unittest.TestCase):
    def test_load_example(self) -> None:
        ds = ms.load(EXAMPLE)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.n_peaks_total, 7)
        self.assertEqual(ds.columns, ["NAME", "PRECURSORMZ", "PRECURSORTYPE", "FORMULA",
                                      "COLLISIONENERGY", "Num Peaks"])
        self.assertEqual(ds["NAME"], ["MassSpecID0000001", "MassSpecID0000002"])
        self.assertEqual(ds.value(1, "COLLISIONENERGY"), "20.0")

    def test_all_formats_and_reference_files(self) -> None:
        for name, size in [("sample.msp", 5), ("sample.mgf", 6), ("sample.msds", 6)]:
            with self.subTest(name=name):
                self.assertEqual(len(ms.load(DATA / name)), size)
        ds = ms.load(DATA / "sample.msp", spec_id_prefix="SP")
        self.assertEqual(ds["SpecID"][:2], ["SP1", "SP2"])

    def test_read_text_and_type_inference(self) -> None:
        text = "Name\tID\tPeak\nA\t007\t1,2\n"
        self.assertEqual(ms.read_text(text, "tsv")["ID"], ["007"])
        self.assertEqual(ms.read_text(text, "tsv", infer_types=True)["ID"], [7])
        mgf = ms.read_text("BEGIN IONS\nTITLE=x\n100 1\nEND IONS\n", "mgf")
        self.assertEqual(mgf["TITLE"], ["x"])

    def test_load_text_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "duplicate.msp"
            source.write_text(
                "Name: first\nNAME: alias\nNumPeaks: 1\n10 50\n\n", encoding="utf-8")
            self.assertEqual(len(ms.load(source, canonicalize_column=True)), 0)
            ds = ms.load(source, canonicalize_column=True, allow_duplicate_cols=True,
                         normalize_intensity=True)
            self.assertEqual(len(ds), 1)
            self.assertEqual(ds.value(0, "Name"), '"first","alias"')
            self.assertNotIn("Name1", ds.columns)
            self.assertEqual(ds[0].intensity.tolist(), [1.0])

            log = Path(directory) / "read-errors.txt"
            ms.load(source, canonicalize_column=True, error_log_level="basic",
                    error_log_file=log)
            self.assertIn("Duplicate meta key", log.read_text(encoding="utf-8"))

        with self.assertRaises(ValueError):
            ms.load(EXAMPLE, error_log_level="verbose")

    def test_errors(self) -> None:
        with self.assertRaises(ms.FileError):
            ms.load("/nonexistent/file.msds")
        with self.assertRaises(OSError):          # FileError is an OSError
            ms.load("/nonexistent/file.msp")
        ds = ms.load(EXAMPLE)
        with self.assertRaises(KeyError):
            ds["Missing"]
        with self.assertRaises(IndexError):
            ds[5]
        with self.assertRaises(ValueError):
            ds.set_column("Bad", [1, 2, 3])
        with self.assertRaises(ValueError):
            ms.read_text("x", "xml")


class TestRecordsAndPeaks(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = ms.load(EXAMPLE)

    def test_record(self) -> None:
        rec = self.ds[0]
        self.assertEqual(rec["NAME"], "MassSpecID0000001")
        self.assertEqual(rec.n_peaks, 4)
        np.testing.assert_allclose(rec.mz, [91.0542, 125.0233, 185.0961, 246.1125])
        np.testing.assert_allclose(rec.intensity, [245, 1000, 349, 735])
        self.assertEqual(self.ds[-1]["NAME"], "MassSpecID0000002")
        self.assertEqual([r["NAME"] for r in self.ds], self.ds["NAME"])
        self.assertEqual(rec.to_dict()["Num Peaks"], "4")

    def test_all_peaks(self) -> None:
        np.testing.assert_array_equal(self.ds.offsets, [0, 4, 7])
        self.assertEqual(self.ds.mz.shape, (7,))
        self.assertAlmostEqual(self.ds.intensity[4], 99.0)

    def test_peak_metadata(self) -> None:
        self.assertEqual(self.ds.peak_columns, ["formula"])
        self.assertEqual(self.ds.peak_column("formula")[:2], ["C7H7+", "C6H5O3+"])
        self.ds.set_peak_column("note", ["n"] * 7)
        self.assertEqual(self.ds.peak_columns, ["formula", "note"])

    def test_normalize_and_sort(self) -> None:
        self.ds.normalize(100.0)
        self.ds.sort_peaks("intensity")
        np.testing.assert_allclose(self.ds[0].intensity, [100.0, 73.5, 34.9, 24.5])
        self.ds.sort_peaks("mz")
        self.assertTrue(np.all(np.diff(self.ds[0].mz) > 0))


class TestViews(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = ms.load(EXAMPLE)

    def test_indexing_returns_views(self) -> None:
        self.assertEqual(self.ds[1:]["NAME"], ["MassSpecID0000002"])
        self.assertEqual(self.ds[[1, 0]]["NAME"], ["MassSpecID0000002", "MassSpecID0000001"])
        self.assertEqual(self.ds[np.array([False, True])]["NAME"], ["MassSpecID0000002"])
        self.assertEqual(self.ds[::-1]["NAME"], ["MassSpecID0000002", "MassSpecID0000001"])
        self.assertEqual(self.ds.sort_by("COLLISIONENERGY")["NAME"][0], "MassSpecID0000002")

    def test_views_share_data_and_copy_does_not(self) -> None:
        view = self.ds[[1]]
        view["Checked"] = True
        self.assertNotIn("Checked", self.ds)                 # own column list
        self.assertEqual(self.ds.full_view(reset_columns=True)["Checked"], [None, True])
        copy = view.copy()
        copy["Checked"] = False
        self.assertEqual(self.ds.full_view(reset_columns=True)["Checked"], [None, True])
        np.testing.assert_array_equal(copy.offsets, [0, 3])

    def test_view_outlives_original(self) -> None:
        view = self.ds[[0]]
        self.ds.close()
        del self.ds
        self.assertEqual(view["NAME"], ["MassSpecID0000001"])


class TestEditingAndSaving(unittest.TestCase):
    def test_build_edit_save_roundtrip(self) -> None:
        ds = ms.from_peaks([[(100.0, 1.0), (150.0, 0.5)], [(90.0, 1.0)]])
        ds["NAME"] = ["a", "b"]
        ds["Score"] = [0.5, float("nan")]
        ds["Count"] = np.array([1, 2])
        ds["Flag"] = True
        ds[1]["NAME"] = "renamed"
        ds.description = "built from Python"
        ds.set_attribute("source", "unit test")
        self.assertTrue(ds.add_tag("python"))
        self.assertFalse(ds.add_tag("python"))
        self.assertTrue(ds.set_spec_id(prefix="SP"))
        ds.set_peak_column("formula", ["X", "", "Y"])
        ds.columns = ["NAME", "Score", "Count", "Flag", "SpecID"]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "built.msds"
            ds.save(path)
            back = ms.load(path)
            self.assertEqual(back["NAME"], ["a", "renamed"])
            self.assertEqual(back["Score"], [0.5, None])
            self.assertEqual(back["Count"], [1, 2])
            self.assertEqual(back["Flag"], [True, True])
            self.assertEqual(back["SpecID"], ["SP1", "SP2"])
            self.assertEqual(back.description, "built from Python")
            self.assertEqual(back.attributes, {"source": "unit test"})
            self.assertEqual(back.tags, ["python"])
            self.assertEqual(back.peak_column("formula"), ["X", "", "Y"])
            for ext in ("msp", "mgf", "tsv", "csv"):
                out = Path(tmp) / f"built.{ext}"
                ds.save(out)
                self.assertEqual(len(ms.load(out)), 2, ext)

    def test_bytes_round_trip(self) -> None:
        ds = ms.load(EXAMPLE)[[1]]
        data = ds.to_bytes()
        self.assertEqual(data[1:4], b"HDF")
        back = ms.from_bytes(data)
        self.assertEqual(back["NAME"], ["MassSpecID0000002"])
        self.assertEqual(back.peak_column("formula"), ["C7H7+", "", ""])
        with self.assertRaises(ms.FormatError):
            ms.from_bytes(b"not an hdf5 file")

    def test_concat(self) -> None:
        a = ms.load(EXAMPLE)
        b = ms.from_peaks([[(1.0, 1.0)]])
        joined = ms.concat([a, b])
        self.assertEqual(len(joined), 3)
        self.assertEqual(joined["NAME"][2], None)

    def test_arrow_and_pandas(self) -> None:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        ds = ms.load(EXAMPLE)
        table = ds.to_arrow()
        self.assertEqual(table.num_rows, 2)
        self.assertEqual(table.column_names, ds.columns)
        self.assertEqual(ds.peak_metadata_to_arrow().column("formula").to_pylist()[0], "C7H7+")
        frame = ds.to_pandas()
        self.assertEqual(frame["NAME"].tolist(), ds["NAME"])


if __name__ == "__main__":
    unittest.main()
