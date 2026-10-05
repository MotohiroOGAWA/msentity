from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from msentity import MSDataset, PeakSeries

ROOT = Path(__file__).resolve().parents[2]
BACKEND_PYTHON = ROOT / "vscode-extension" / "python"
sys.path.insert(0, str(BACKEND_PYTHON))
backend = importlib.import_module("backend")
chemistry_module = importlib.import_module("chemistry")


class FakeQuery:
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern


class FakeMolecule:
    def __init__(self, smiles: str) -> None:
        self.smiles = smiles

    def HasSubstructMatch(self, query: FakeQuery) -> bool:
        if query.pattern == "C":
            return "C" in self.smiles
        if query.pattern == "c1ccccc1":
            return "c1ccccc1" in self.smiles
        if query.pattern == "[#6](=O)[#8]":
            return "C(=O)O" in self.smiles
        return False


class FakeChem:
    def __init__(self) -> None:
        self.smiles_calls: list[str] = []
        self.smarts_calls: list[str] = []

    def MolFromSmiles(self, smiles: str):
        self.smiles_calls.append(smiles)
        return None if smiles in {"", "invalid smiles"} else FakeMolecule(smiles)

    def MolFromSmarts(self, pattern: str):
        self.smarts_calls.append(pattern)
        return None if pattern == "[invalid" else FakeQuery(pattern)


class FakeDrawer:
    class MolDraw2DSVG:
        def __init__(self, width: int, height: int) -> None:
            self.molecule = None

        def DrawMolecule(self, molecule) -> None:
            self.molecule = molecule

        def FinishDrawing(self) -> None:
            pass

        def GetDrawingText(self) -> str:
            return f"<svg><text>{self.molecule.smiles}</text></svg>"


def fake_backend() -> tuple[chemistry_module.ChemistryBackend, FakeChem]:
    chem = FakeChem()
    modules = {
        "rdkit": object(),
        "rdkit.Chem": chem,
        "rdkit.Chem.Draw.rdMolDraw2D": FakeDrawer,
    }
    return chemistry_module.ChemistryBackend(modules.__getitem__), chem


def dataset_with_structures(values: list[str], column: str = "smiles") -> MSDataset:
    peaks = PeakSeries(np.empty((0, 2)), np.zeros(len(values) + 1, dtype=np.int64))
    return MSDataset(pd.DataFrame({column: values, "order": range(len(values))}), peaks)


class ChemistryCapabilityTest(unittest.TestCase):
    def test_environment_can_force_rdkit_off_for_extension_debugging(self) -> None:
        old = os.environ.get("MSENTITY_SPECTRUM_VIEWER_DISABLE_RDKIT")
        os.environ["MSENTITY_SPECTRUM_VIEWER_DISABLE_RDKIT"] = "1"
        try:
            chemistry, _ = fake_backend()
        finally:
            if old is None:
                os.environ.pop("MSENTITY_SPECTRUM_VIEWER_DISABLE_RDKIT", None)
            else:
                os.environ["MSENTITY_SPECTRUM_VIEWER_DISABLE_RDKIT"] = old
        self.assertEqual(chemistry.capabilities, {
            "backend": None, "smarts_filter": False, "structure_render": False,
        })

    def test_missing_rdkit_disables_only_chemistry_capabilities(self) -> None:
        def missing(_name: str):
            raise ModuleNotFoundError("rdkit")

        chemistry = chemistry_module.ChemistryBackend(missing)
        self.assertEqual(chemistry.capabilities, {
            "backend": None, "smarts_filter": False, "structure_render": False,
        })

    def test_smoke_tests_enable_features_independently(self) -> None:
        chemistry, _ = fake_backend()
        self.assertTrue(chemistry.capabilities["smarts_filter"])
        self.assertTrue(chemistry.capabilities["structure_render"])
        self.assertTrue(all(chemistry.smoke_tests.values()))

        chem = FakeChem()
        modules = {"rdkit": object(), "rdkit.Chem": chem}
        partial = chemistry_module.ChemistryBackend(modules.__getitem__)
        self.assertTrue(partial.capabilities["smarts_filter"])
        self.assertFalse(partial.capabilities["structure_render"])

        no_smarts = FakeChem()
        no_smarts.MolFromSmarts = lambda _pattern: None
        modules = {
            "rdkit": object(), "rdkit.Chem": no_smarts,
            "rdkit.Chem.Draw.rdMolDraw2D": FakeDrawer,
        }
        partial = chemistry_module.ChemistryBackend(modules.__getitem__)
        self.assertFalse(partial.capabilities["smarts_filter"])
        self.assertTrue(partial.capabilities["structure_render"])

    def test_smiles_cache_and_per_query_match_cache_use_smiles_keys(self) -> None:
        chemistry, chem = fake_backend()
        baseline = len(chem.smiles_calls)
        matches = chemistry.smarts_matches(["CCO", "CCO", "invalid smiles", "invalid smiles"], "C")
        self.assertEqual(matches, [True, True, False, False])
        self.assertEqual(chem.smiles_calls[baseline:], ["CCO", "invalid smiles"])
        self.assertEqual(chem.smarts_calls[-1], "C")
        self.assertEqual(set(chemistry._molecule_cache), {"CCO", "invalid smiles"})
        chemistry.render_svg("CCO")
        self.assertEqual(chem.smiles_calls[baseline:], ["CCO", "invalid smiles"])

    def test_invalid_smarts_and_structure_are_recoverable(self) -> None:
        chemistry, _ = fake_backend()
        with self.assertRaisesRegex(chemistry_module.ChemistryError, "Invalid SMARTS"):
            chemistry.smarts_matches(["CCO"], "[invalid")
        with self.assertRaisesRegex(chemistry_module.ChemistryError, "Invalid SMILES"):
            chemistry.render_svg("invalid smiles")

    def test_smarts_filter_runs_before_pagination_and_supports_any_column(self) -> None:
        chemistry, _ = fake_backend()
        values = ["CC(=O)O" if i % 2 == 0 else "N" for i in range(100)]
        dataset = dataset_with_structures(values, "my_structure")
        view = backend.apply_view(
            dataset,
            [{"column": "my_structure", "operator": "smarts", "value": "[#6](=O)[#8]"}],
            [{"column": "order", "direction": "desc"}],
            chemistry=chemistry,
        )
        page = backend.serialize_page(view, 0, 20, "dataset", Path("sample.msds"))
        self.assertEqual(page["total_rows"], 50)
        self.assertEqual(page["total_pages"], 3)
        self.assertEqual(len(page["rows"]), 20)
        self.assertEqual(page["rows"][0]["order"], 98)

    def test_smiles_column_detection_priority_is_case_insensitive(self) -> None:
        detect = chemistry_module.detect_smiles_column
        self.assertEqual(detect(["ISOMERIC_SMILES", "Canonical_SMILES", "SMILES"]), "SMILES")
        self.assertEqual(detect(["Name", "Canonical_SMILES"]), "Canonical_SMILES")
        self.assertIsNone(detect(["Name", "Formula"]))

    def test_backend_starts_and_reports_capabilities_without_breaking_core_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.tsv"
            source.write_text("Name\tPeak\na\t100,10\nb\t200,20\n")
            result = subprocess.run(
                [sys.executable, str(BACKEND_PYTHON / "backend.py"), str(source), "--page-size", "1"],
                input=json.dumps({"type": "page", "page": 1,
                                  "filters": [{"column": "Name", "operator": "text_eq", "value": "b"}],
                                  "sort": [{"column": "Name", "direction": "desc"}]}) + "\n",
                text=True, capture_output=True, timeout=30, check=True,
                env=dict(os.environ, PYTHONPATH=str(ROOT)),
            )
        messages = [json.loads(line.split("MSENTITY_JSON:", 1)[1])
                    for line in result.stdout.splitlines() if line.startswith("MSENTITY_JSON:")]
        capabilities = next(message["chemistry"] for message in messages if message["type"] == "capabilities")
        if importlib.util.find_spec("rdkit") is None:
            self.assertEqual(capabilities, {
                "backend": None, "smarts_filter": False, "structure_render": False,
            })
        else:
            self.assertTrue(capabilities["smarts_filter"])
            self.assertTrue(capabilities["structure_render"])
        page = next(message["value"] for message in messages if message["type"] == "dataset-page")
        self.assertEqual(page["total_rows"], 1)
        self.assertEqual(page["rows"][0]["Name"], "b")


@unittest.skipUnless(importlib.util.find_spec("rdkit"), "RDKit optional dependency is not installed")
class RealRDKitTest(unittest.TestCase):
    def test_parse_match_invalid_inputs_and_svg(self) -> None:
        chemistry = chemistry_module.ChemistryBackend()
        self.assertTrue(chemistry.capabilities["smarts_filter"])
        self.assertTrue(chemistry.capabilities["structure_render"])
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        self.assertEqual(chemistry.smarts_matches([aspirin], "c1ccccc1"), [True])
        self.assertEqual(chemistry.smarts_matches([aspirin], "[N+](=O)[O-]"), [False])
        self.assertEqual(chemistry.smarts_matches(["invalid smiles"], "C"), [False])
        with self.assertRaises(chemistry_module.ChemistryError):
            chemistry.smarts_matches([aspirin], "[invalid")
        smiles, svg = chemistry.render_svg(aspirin)
        self.assertEqual(smiles, aspirin)
        self.assertIn("<svg", svg)

    def test_backend_smarts_and_structure_protocol(self) -> None:
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "structures.tsv"
            source.write_text(
                "Name\tmy_structure\tPeak\n"
                f"aspirin\t{aspirin}\t100,10\n"
                "ethanol\tCCO\t200,20\n"
            )
            requests = [
                {"type": "page", "filters": [{
                    "column": "my_structure", "operator": "smarts", "value": "c1ccccc1",
                }]},
                {"type": "render-structure", "smiles": aspirin},
                {"type": "page", "filters": [{
                    "column": "my_structure", "operator": "smarts", "value": "[invalid",
                }]},
            ]
            result = subprocess.run(
                [sys.executable, str(BACKEND_PYTHON / "backend.py"), str(source)],
                input="".join(json.dumps(request) + "\n" for request in requests),
                text=True, capture_output=True, timeout=30, check=True,
                env=dict(os.environ, PYTHONPATH=str(ROOT)),
            )
        messages = [json.loads(line.split("MSENTITY_JSON:", 1)[1])
                    for line in result.stdout.splitlines() if line.startswith("MSENTITY_JSON:")]
        page = next(message["value"] for message in messages if message["type"] == "dataset-page")
        self.assertEqual(page["total_rows"], 1)
        self.assertEqual(page["rows"][0]["Name"], "aspirin")
        rendered = next(message for message in messages if message["type"] == "structure-rendered")
        self.assertEqual(rendered["smiles"], aspirin)
        self.assertIn("<svg", rendered["svg"])
        error = next(message for message in messages if message["type"] == "filter-error")
        self.assertEqual(error["operator"], "smarts")


if __name__ == "__main__":
    unittest.main()
