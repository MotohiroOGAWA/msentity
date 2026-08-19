from __future__ import annotations

import unittest
from argparse import Namespace
from unittest.mock import patch

from msentity import load_ms_dataset
from msentity.cli.gui import run
from msentity.cli.gui.app import spectrum_view
from msentity.cli.main import build_parser
from tests.common import SAMPLE_MGF_FILE, SAMPLE_MSP_FILE


class TestGUI(unittest.TestCase):
    def test_gui_parser_options(self) -> None:
        args = build_parser().parse_args([
            "gui",
            str(SAMPLE_MSP_FILE),
            "--file-type",
            "msp",
            "--spec-id-prefix",
            "test",
            "--host",
            "0.0.0.0",
            "--port",
            "7861",
            "--share",
            "--no-browser",
        ])
        self.assertEqual(args.command, "gui")
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 7861)
        self.assertTrue(args.share)
        self.assertTrue(args.no_browser)

    def test_gui_discovery_does_not_import_optional_dependencies(self) -> None:
        with patch("builtins.__import__", wraps=__import__) as import_mock:
            build_parser()
        imported = {call.args[0] for call in import_mock.call_args_list if call.args}
        self.assertNotIn("gradio", imported)
        self.assertNotIn("gradio_msentityviewer", imported)

    def test_selected_spectrum_adapter_for_msp_and_mgf(self) -> None:
        for input_file in (SAMPLE_MSP_FILE, SAMPLE_MGF_FILE):
            with self.subTest(input_file=input_file):
                dataset = load_ms_dataset(input_file)
                selected, label, value, metadata = spectrum_view(dataset, 0)
                self.assertEqual(selected, 0)
                self.assertEqual(label, f"Spectrum 1 / {len(dataset)}")
                self.assertEqual(len(value["mz"]), len(value["intensity"]))
                self.assertEqual(set(metadata), set(dataset.columns))

    def test_selection_is_clamped(self) -> None:
        dataset = load_ms_dataset(SAMPLE_MSP_FILE)
        self.assertEqual(spectrum_view(dataset, -1)[0], 0)
        self.assertEqual(spectrum_view(dataset, len(dataset) + 1)[0], len(dataset) - 1)

    def test_missing_gui_dependency_has_install_hint(self) -> None:
        args = Namespace(
            input_file=str(SAMPLE_MSP_FILE),
            file_type=None,
            spec_id_prefix=None,
            host="127.0.0.1",
            port=7860,
            share=False,
            no_browser=True,
        )
        with patch(
            "msentity.cli.gui.app.run_gui",
            side_effect=ImportError("missing optional dependency"),
        ):
            with self.assertRaisesRegex(SystemExit, r'msentity\[gui\]'):
                run(args)


if __name__ == "__main__":
    unittest.main()
