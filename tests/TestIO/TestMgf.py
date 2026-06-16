import os
import tempfile
import unittest

from msentity.io.mgf import read_mgf, read_mgf_text, write_mgf


class TestMGFIO(unittest.TestCase):
    """Unit tests for MGF reader and writer."""

    def _write_temp_file(self, text: str, suffix: str = ".mgf") -> str:
        """Create a temporary text file and return its path."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def _read_text(self, path: str) -> str:
        """Read a text file."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_read_mgf_basic(self) -> None:
        """read_mgf should read a simple MGF file."""
        mgf_text = (
            "BEGIN IONS\n"
            "Name=spec1\n"
            "PEPMASS=100.0\n"
            "SMILES=CCO\n"
            "50.0\t10.0\n"
            "60.0\t20.0\n"
            "70.0\t30.0\n"
            "END IONS\n"
            "\n"
            "BEGIN IONS\n"
            "Name=spec2\n"
            "PEPMASS=200.0\n"
            "SMILES=CCC\n"
            "80.0\t40.0\n"
            "90.0\t50.0\n"
            "END IONS\n"
        )
        path = self._write_temp_file(mgf_text)

        try:
            dataset = read_mgf(path, show_progress=False)

            self.assertEqual(len(dataset), 2)
            self.assertIn("Name", dataset.columns)
            self.assertIn("PEPMASS", dataset.columns)
            self.assertIn("SMILES", dataset.columns)

            self.assertEqual(str(dataset.metadata.iloc[0]["Name"]), "spec1")
            self.assertEqual(str(dataset.metadata.iloc[1]["Name"]), "spec2")
            self.assertEqual(dataset[0].n_peaks, 3)
            self.assertEqual(dataset[1].n_peaks, 2)
        finally:
            os.remove(path)

    def test_read_mgf_without_trailing_blank_line(self) -> None:
        """read_mgf should handle EOF without trailing blank lines."""
        mgf_text = (
            "BEGIN IONS\n"
            "Name=spec1\n"
            "PEPMASS=100.0\n"
            "50.0\t10.0\n"
            "60.0\t20.0\n"
            "END IONS\n"
        )
        path = self._write_temp_file(mgf_text)

        try:
            dataset = read_mgf(path, show_progress=False)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0].n_peaks, 2)
            self.assertEqual(str(dataset.metadata.iloc[0]["Name"]), "spec1")
        finally:
            os.remove(path)

    def test_read_mgf_return_header_map(self) -> None:
        """read_mgf should return header_map when requested."""
        mgf_text = (
            "BEGIN IONS\n"
            "Name=spec1\n"
            "Comment=hello\n"
            "50.0\t10.0\n"
            "END IONS\n"
        )
        path = self._write_temp_file(mgf_text)

        try:
            dataset, header_map = read_mgf(
                path,
                return_header_map=True,
                show_progress=False,
            )

            self.assertEqual(len(dataset), 1)
            self.assertIsInstance(header_map, dict)
            self.assertIn("Name", header_map)
            self.assertEqual(header_map["Name"], "Name")
        finally:
            os.remove(path)

    def test_read_mgf_with_spec_id_prefix(self) -> None:
        """read_mgf should assign spectrum IDs when spec_id_prefix is given."""
        mgf_text = (
            "BEGIN IONS\n"
            "Name=spec1\n"
            "50.0\t10.0\n"
            "END IONS\n"
            "\n"
            "BEGIN IONS\n"
            "Name=spec2\n"
            "60.0\t20.0\n"
            "END IONS\n"
        )
        path = self._write_temp_file(mgf_text)

        try:
            dataset = read_mgf(path, spec_id_prefix="SP", show_progress=False)

            self.assertIn("SpecID", dataset.columns)
            spec_ids = dataset["SpecID"].astype(str).tolist()
            self.assertEqual(len(spec_ids), 2)
            self.assertTrue(all(v.startswith("SP") for v in spec_ids))
        finally:
            os.remove(path)

    def test_read_mgf_with_header_peak_columns(self) -> None:
        """read_mgf should parse a peak header row and attach peak metadata."""
        mgf_text = (
            "BEGIN IONS\n"
            "Name=spec1\n"
            "mz intensity annotation note\n"
            "50.0 10.0 fragA ; noteA\n"
            "60.0 20.0 fragB ; noteB\n"
            "END IONS\n"
        )
        path = self._write_temp_file(mgf_text)

        try:
            dataset = read_mgf(path, show_progress=False)

            self.assertIsNotNone(dataset.peaks.metadata)
            self.assertIn("annotation", dataset.peaks.metadata.columns)
            self.assertIn("note", dataset.peaks.metadata.columns)
            self.assertEqual(
                dataset.peaks.metadata["annotation"].astype(str).tolist(),
                ["fragA", "fragB"],
            )
            self.assertEqual(
                dataset.peaks.metadata["note"].astype(str).tolist(),
                ["noteA", "noteB"],
            )
        finally:
            os.remove(path)

    def test_read_mgf_with_custom_peak_parser(self) -> None:
        """read_mgf should use a custom single-line peak parser when provided."""
        mgf_text = (
            "BEGIN IONS\n"
            "Name=spec1\n"
            "50.0\t10.0\n"
            "60.0\t20.0\n"
            "END IONS\n"
        )
        path = self._write_temp_file(mgf_text)

        def custom_peak_parser(line: str):
            items = line.strip().split("\t")
            return {
                "mz": float(items[0]),
                "intensity": float(items[1]),
                "label": "custom",
            }

        try:
            dataset = read_mgf(
                path,
                peak_parser=custom_peak_parser,
                show_progress=False,
            )

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0].n_peaks, 2)
            self.assertIsNotNone(dataset.peaks.metadata)
            self.assertIn("label", dataset.peaks.metadata.columns)
            self.assertEqual(
                dataset.peaks.metadata["label"].astype(str).tolist(),
                ["custom", "custom"],
            )
        finally:
            os.remove(path)

    def test_read_mgf_duplicate_meta_raises_record_error_and_skips_record(self) -> None:
        """A record with duplicate metadata keys should be skipped when duplicates are not allowed."""
        mgf_text = (
            "BEGIN IONS\n"
            "Name=spec1\n"
            "Name=spec1_dup\n"
            "50.0\t10.0\n"
            "END IONS\n"
            "\n"
            "BEGIN IONS\n"
            "Name=spec2\n"
            "60.0\t20.0\n"
            "END IONS\n"
        )
        path = self._write_temp_file(mgf_text)

        try:
            dataset = read_mgf(path, allow_duplicate_cols=False, show_progress=False)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(str(dataset.metadata.iloc[0]["Name"]), "spec2")
        finally:
            os.remove(path)

    def test_write_mgf_basic(self) -> None:
        """write_mgf should write a readable MGF file."""
        mgf_text = (
            "BEGIN IONS\n"
            "Name=spec1\n"
            "PEPMASS=100.0\n"
            "SMILES=CCO\n"
            "50.0\t10.0\n"
            "60.0\t20.0\n"
            "END IONS\n"
            "\n"
            "BEGIN IONS\n"
            "Name=spec2\n"
            "PEPMASS=200.0\n"
            "SMILES=CCC\n"
            "80.0\t30.0\n"
            "END IONS\n"
        )
        in_path = self._write_temp_file(mgf_text)
        out_fd, out_path = tempfile.mkstemp(suffix=".mgf")
        os.close(out_fd)

        try:
            dataset = read_mgf(in_path, show_progress=False)
            write_mgf(dataset, out_path, show_progress=False)

            written = self._read_text(out_path)

            self.assertIn("BEGIN IONS", written)
            self.assertIn("END IONS", written)
            self.assertIn("Name=spec1", written)
            self.assertIn("Name=spec2", written)
            self.assertIn("50.0\t0.5", written)
            self.assertIn("80.0\t1.0", written)
        finally:
            os.remove(in_path)
            os.remove(out_path)

    def test_write_mgf_with_header_map(self) -> None:
        """write_mgf should rename spectrum headers using header_map."""
        mgf_text = (
            "BEGIN IONS\n"
            "Name=spec1\n"
            "PEPMASS=100.0\n"
            "50.0\t10.0\n"
            "END IONS\n"
        )
        in_path = self._write_temp_file(mgf_text)
        out_fd, out_path = tempfile.mkstemp(suffix=".mgf")
        os.close(out_fd)

        try:
            dataset = read_mgf(in_path, show_progress=False)
            write_mgf(
                dataset,
                out_path,
                headers=["Name", "PEPMASS"],
                header_map={"Name": "TITLE", "PEPMASS": "PRECURSOR_MZ"},
                show_progress=False,
            )

            written = self._read_text(out_path)
            self.assertIn("TITLE=spec1", written)
            self.assertIn("PRECURSOR_MZ=100.0", written)
            self.assertNotIn("Name=spec1", written)
        finally:
            os.remove(in_path)
            os.remove(out_path)

    def test_write_mgf_with_selected_headers(self) -> None:
        """write_mgf should write only selected spectrum headers."""
        mgf_text = (
            "BEGIN IONS\n"
            "Name=spec1\n"
            "PEPMASS=100.0\n"
            "SMILES=CCO\n"
            "50.0\t10.0\n"
            "END IONS\n"
        )
        in_path = self._write_temp_file(mgf_text)
        out_fd, out_path = tempfile.mkstemp(suffix=".mgf")
        os.close(out_fd)

        try:
            dataset = read_mgf(in_path, show_progress=False)
            write_mgf(
                dataset,
                out_path,
                headers=["Name"],
                show_progress=False,
            )

            written = self._read_text(out_path)
            self.assertIn("Name=spec1", written)
            self.assertNotIn("PEPMASS=", written)
            self.assertNotIn("SMILES=", written)
        finally:
            os.remove(in_path)
            os.remove(out_path)

    def test_write_mgf_with_peak_headers(self) -> None:
        """write_mgf should write peak metadata headers and values when requested."""
        mgf_text = (
            "BEGIN IONS\n"
            "Name=spec1\n"
            "mz intensity annotation note\n"
            "50.0 10.0 fragA ; noteA\n"
            "60.0 20.0 fragB ; noteB\n"
            "END IONS\n"
        )
        in_path = self._write_temp_file(mgf_text)
        out_fd, out_path = tempfile.mkstemp(suffix=".mgf")
        os.close(out_fd)

        try:
            dataset = read_mgf(in_path, show_progress=False)
            write_mgf(
                dataset,
                out_path,
                peak_headers=["annotation", "note"],
                show_progress=False,
            )

            written = self._read_text(out_path)
            self.assertIn("mz\tintensity\tannotation\tnote", written)
            self.assertIn('50.0\t0.5\t"fragA" ; "noteA"', written)
            self.assertIn('60.0\t1.0\t"fragB" ; "noteB"', written)
        finally:
            os.remove(in_path)
            os.remove(out_path)

    def test_write_mgf_round_trip(self) -> None:
        """An MGF written by write_mgf should be readable again by read_mgf."""
        mgf_text = (
            "BEGIN IONS\n"
            "Name=spec1\n"
            "PEPMASS=100.0\n"
            "SMILES=CCO\n"
            "50.0\t10.0\n"
            "60.0\t20.0\n"
            "70.0\t30.0\n"
            "END IONS\n"
            "\n"
            "BEGIN IONS\n"
            "Name=spec2\n"
            "PEPMASS=200.0\n"
            "SMILES=CCC\n"
            "80.0\t40.0\n"
            "END IONS\n"
        )
        in_path = self._write_temp_file(mgf_text)
        out_fd, out_path = tempfile.mkstemp(suffix=".mgf")
        os.close(out_fd)

        try:
            dataset1 = read_mgf(in_path, show_progress=False)
            write_mgf(dataset1, out_path, show_progress=False)
            dataset2 = read_mgf(out_path, show_progress=False)

            self.assertEqual(len(dataset1), len(dataset2))
            self.assertEqual(dataset1[0].n_peaks, dataset2[0].n_peaks)
            self.assertEqual(dataset1[1].n_peaks, dataset2[1].n_peaks)
            self.assertEqual(
                str(dataset1.metadata.iloc[0]["Name"]),
                str(dataset2.metadata.iloc[0]["Name"]),
            )
            self.assertEqual(
                str(dataset1.metadata.iloc[1]["Name"]),
                str(dataset2.metadata.iloc[1]["Name"]),
            )
        finally:
            os.remove(in_path)
            os.remove(out_path)

    def test_read_mgf_text_basic(self) -> None:
        """read_mgf_text should read simple MGF text."""
        mgf_text = (
            "BEGIN IONS\n"
            "TITLE=spec1\n"
            "PEPMASS=100.0\n"
            "CHARGE=1+\n"
            "50.0\t10.0\n"
            "60.0\t20.0\n"
            "END IONS\n"
            "\n"
            "BEGIN IONS\n"
            "TITLE=spec2\n"
            "PEPMASS=200.0\n"
            "CHARGE=1+\n"
            "80.0\t40.0\n"
            "90.0\t50.0\n"
            "END IONS\n"
        )

        dataset = read_mgf_text(
            mgf_text,
            show_progress=False,
        )

        self.assertEqual(len(dataset), 2)
        self.assertIn("TITLE", dataset.columns)
        self.assertIn("PEPMASS", dataset.columns)
        self.assertIn("CHARGE", dataset.columns)

        self.assertEqual(str(dataset.metadata.iloc[0]["TITLE"]), "spec1")
        self.assertEqual(str(dataset.metadata.iloc[1]["TITLE"]), "spec2")
        self.assertEqual(dataset[0].n_peaks, 2)
        self.assertEqual(dataset[1].n_peaks, 2)

    def test_read_mgf_text_matches_read_mgf(self) -> None:
        """read_mgf_text should produce the same result as read_mgf."""
        mgf_text = (
            "BEGIN IONS\n"
            "TITLE=spec1\n"
            "PEPMASS=100.0\n"
            "CHARGE=1+\n"
            "50.0\t10.0\n"
            "60.0\t20.0\n"
            "END IONS\n"
            "\n"
            "BEGIN IONS\n"
            "TITLE=spec2\n"
            "PEPMASS=200.0\n"
            "CHARGE=1+\n"
            "80.0\t40.0\n"
            "END IONS\n"
        )
        path = self._write_temp_file(mgf_text, suffix=".mgf")

        try:
            dataset_from_file = read_mgf(
                path,
                show_progress=False,
            )
            dataset_from_text = read_mgf_text(
                mgf_text,
                show_progress=False,
            )

            self.assertEqual(len(dataset_from_file), len(dataset_from_text))
            self.assertEqual(
                dataset_from_file.metadata["TITLE"].astype(str).tolist(),
                dataset_from_text.metadata["TITLE"].astype(str).tolist(),
            )
            self.assertEqual(dataset_from_file[0].n_peaks, dataset_from_text[0].n_peaks)
            self.assertEqual(dataset_from_file[1].n_peaks, dataset_from_text[1].n_peaks)

            self.assertEqual(
                dataset_from_file.peaks.mz.tolist(),
                dataset_from_text.peaks.mz.tolist(),
            )
            self.assertEqual(
                dataset_from_file.peaks.intensity.tolist(),
                dataset_from_text.peaks.intensity.tolist(),
            )
        finally:
            os.remove(path)

    def test_read_mgf_text_without_end_ions(self) -> None:
        """read_mgf_text should handle EOF without END IONS."""
        mgf_text = (
            "BEGIN IONS\n"
            "TITLE=spec1\n"
            "PEPMASS=100.0\n"
            "CHARGE=1+\n"
            "50.0\t10.0\n"
            "60.0\t20.0\n"
        )

        dataset = read_mgf_text(
            mgf_text,
            show_progress=False,
        )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].n_peaks, 2)
        self.assertEqual(str(dataset.metadata.iloc[0]["TITLE"]), "spec1")

    def test_read_mgf_text_return_header_map(self) -> None:
        """read_mgf_text should return header_map when requested."""
        mgf_text = (
            "BEGIN IONS\n"
            "TITLE=spec1\n"
            "PEPMASS=100.0\n"
            "50.0\t10.0\n"
            "END IONS\n"
        )

        dataset, header_map = read_mgf_text(
            mgf_text,
            return_header_map=True,
            show_progress=False,
        )

        self.assertEqual(len(dataset), 1)
        self.assertIsInstance(header_map, dict)
        self.assertIn("TITLE", header_map)
        self.assertEqual(header_map["TITLE"], "TITLE")

    def test_read_mgf_text_with_spec_id_prefix(self) -> None:
        """read_mgf_text should assign spectrum IDs when spec_id_prefix is given."""
        mgf_text = (
            "BEGIN IONS\n"
            "TITLE=spec1\n"
            "50.0\t10.0\n"
            "END IONS\n"
            "\n"
            "BEGIN IONS\n"
            "TITLE=spec2\n"
            "60.0\t20.0\n"
            "END IONS\n"
        )

        dataset = read_mgf_text(
            mgf_text,
            spec_id_prefix="MGF",
            show_progress=False,
        )

        self.assertIn("SpecID", dataset.columns)
        spec_ids = dataset["SpecID"].astype(str).tolist()
        self.assertEqual(len(spec_ids), 2)
        self.assertTrue(all(v.startswith("MGF") for v in spec_ids))

    def test_read_mgf_text_with_header_peak_columns(self) -> None:
        """read_mgf_text should parse a peak header row and attach peak metadata."""
        mgf_text = (
            "BEGIN IONS\n"
            "TITLE=spec1\n"
            "mz intensity annotation note\n"
            "50.0 10.0 fragA ; noteA\n"
            "60.0 20.0 fragB ; noteB\n"
            "END IONS\n"
        )

        dataset = read_mgf_text(
            mgf_text,
            show_progress=False,
        )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].n_peaks, 2)
        self.assertIsNotNone(dataset.peaks.metadata)
        self.assertIn("annotation", dataset.peaks.metadata.columns)
        self.assertIn("note", dataset.peaks.metadata.columns)
        self.assertEqual(
            dataset.peaks.metadata["annotation"].astype(str).tolist(),
            ["fragA", "fragB"],
        )
        self.assertEqual(
            dataset.peaks.metadata["note"].astype(str).tolist(),
            ["noteA", "noteB"],
        )

    def test_read_mgf_text_with_custom_peak_parser(self) -> None:
        """read_mgf_text should use a custom single-line peak parser when provided."""
        mgf_text = (
            "BEGIN IONS\n"
            "TITLE=spec1\n"
            "50.0\t10.0\n"
            "60.0\t20.0\n"
            "END IONS\n"
        )

        def custom_peak_parser(line: str):
            items = line.strip().split("\t")
            return {
                "mz": float(items[0]),
                "intensity": float(items[1]),
                "label": "custom",
            }

        dataset = read_mgf_text(
            mgf_text,
            peak_parser=custom_peak_parser,
            show_progress=False,
        )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].n_peaks, 2)
        self.assertIsNotNone(dataset.peaks.metadata)
        self.assertIn("label", dataset.peaks.metadata.columns)
        self.assertEqual(
            dataset.peaks.metadata["label"].astype(str).tolist(),
            ["custom", "custom"],
        )

    def test_read_mgf_text_ignores_lines_outside_ions_block(self) -> None:
        """read_mgf_text should ignore text outside BEGIN/END IONS blocks."""
        mgf_text = (
            "This line should be ignored\n"
            "Another ignored line\n"
            "BEGIN IONS\n"
            "TITLE=spec1\n"
            "50.0\t10.0\n"
            "END IONS\n"
            "Ignored after block\n"
        )

        dataset = read_mgf_text(
            mgf_text,
            show_progress=False,
        )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].n_peaks, 1)
        self.assertEqual(str(dataset.metadata.iloc[0]["TITLE"]), "spec1")

    def test_read_mgf_text_empty_text_raises(self) -> None:
        """read_mgf_text should raise ValueError for empty text."""
        with self.assertRaises(ValueError):
            read_mgf_text(
                "",
                show_progress=False,
            )

        with self.assertRaises(ValueError):
            read_mgf_text(
                "   \n\n",
                show_progress=False,
            )

if __name__ == "__main__":
    unittest.main()