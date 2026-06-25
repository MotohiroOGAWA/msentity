from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from shutil import copyfile

from msentity import MSDataset
from msentity.cli.main import main
from tests.common import SAMPLE_MSP_FILE


class TestCLI(unittest.TestCase):
    def test_info_command(self) -> None:
        main([
            "info",
            str(SAMPLE_MSP_FILE),
        ])

    def test_head_command(self) -> None:
        main([
            "head",
            str(SAMPLE_MSP_FILE),
            "--num-rows",
            "3",
        ])

    def test_convert_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "dataset.msds"

            main([
                "convert",
                str(SAMPLE_MSP_FILE),
                str(output_file),
            ])

            self.assertTrue(output_file.exists())

    def test_merge_dir_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            copyfile(SAMPLE_MSP_FILE, input_dir / "first.msp")
            copyfile(SAMPLE_MSP_FILE, input_dir / "second.msp")

            output_file = Path(tmpdir) / "merged.msds"

            main([
                "merge-dir",
                str(input_dir),
                str(output_file),
            ])

            self.assertTrue(output_file.exists())
            self.assertEqual(len(MSDataset.load(output_file)), 10)

    def test_merge_dir_recursive_depth_and_source_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            child_dir = input_dir / "child"
            grandchild_dir = child_dir / "grandchild"
            grandchild_dir.mkdir(parents=True)

            copyfile(SAMPLE_MSP_FILE, input_dir / "root.msp")
            copyfile(SAMPLE_MSP_FILE, child_dir / "child.msp")
            copyfile(SAMPLE_MSP_FILE, grandchild_dir / "grandchild.msp")

            output_file = Path(tmpdir) / "merged.msds"

            main([
                "merge-dir",
                str(input_dir),
                str(output_file),
                "--recursive",
                "2",
                "--add-source",
            ])

            dataset = MSDataset.load(output_file)
            self.assertEqual(len(dataset), 10)

            expected_paths = {
                "root.msp",
                "child/child.msp",
            }
            self.assertEqual(set(dataset["path"]), expected_paths)
            self.assertEqual(set(dataset["source_index"]), set(range(5)))

            grouped = dataset.metadata.groupby("path")["source_index"].apply(list).to_dict()
            self.assertEqual(grouped["root.msp"], list(range(5)))
            self.assertEqual(grouped["child/child.msp"], list(range(5)))


if __name__ == "__main__":
    unittest.main()