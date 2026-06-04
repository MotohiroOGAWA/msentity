from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()