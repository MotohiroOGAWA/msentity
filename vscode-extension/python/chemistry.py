from __future__ import annotations

from importlib import import_module
import os
from typing import Any, Callable, Iterable


class ChemistryError(ValueError):
    """A recoverable optional-chemistry operation error."""


class ChemistryBackend:
    """Detect and expose optional RDKit features for one backend process."""

    def __init__(self, importer: Callable[[str], Any] = import_module) -> None:
        self._chem: Any = None
        self._drawer: Any = None
        # Keys are normalized SMILES strings, never dataset or row identifiers.
        self._molecule_cache: dict[str, Any | None] = {}
        self.smoke_tests = {
            "rdkit_import": False,
            "smiles_parse": False,
            "smarts_parse": False,
            "substructure_match": False,
            "svg_drawing": False,
        }
        if os.environ.get("MSENTITY_SPECTRUM_VIEWER_DISABLE_RDKIT") == "1":
            return
        self._detect(importer)

    def _detect(self, importer: Callable[[str], Any]) -> None:
        try:
            importer("rdkit")
            self._chem = importer("rdkit.Chem")
            self.smoke_tests["rdkit_import"] = True
        except Exception:
            return

        molecule = None
        query = None
        try:
            molecule = self._chem.MolFromSmiles("CCO")
            self.smoke_tests["smiles_parse"] = molecule is not None
        except Exception:  # RDKit wrappers may raise implementation-specific errors.
            pass
        try:
            query = self._chem.MolFromSmarts("C")
            self.smoke_tests["smarts_parse"] = query is not None
        except Exception:
            pass
        if molecule is not None and query is not None:
            try:
                self.smoke_tests["substructure_match"] = bool(
                    molecule.HasSubstructMatch(query)
                )
            except Exception:
                pass
        if molecule is not None:
            try:
                self._drawer = importer("rdkit.Chem.Draw.rdMolDraw2D")
                drawer = self._drawer.MolDraw2DSVG(120, 90)
                drawer.DrawMolecule(molecule)
                drawer.FinishDrawing()
                self.smoke_tests["svg_drawing"] = "<svg" in drawer.GetDrawingText()
            except Exception:
                self._drawer = None

    @property
    def capabilities(self) -> dict[str, Any]:
        tests = self.smoke_tests
        smarts_filter = all(
            tests[name]
            for name in ("rdkit_import", "smiles_parse", "smarts_parse", "substructure_match")
        )
        structure_render = all(
            tests[name]
            for name in ("rdkit_import", "smiles_parse", "svg_drawing")
        )
        return {
            "backend": "rdkit" if smarts_filter or structure_render else None,
            "smarts_filter": smarts_filter,
            "structure_render": structure_render,
        }

    @staticmethod
    def normalize_smiles(value: Any) -> str:
        return "" if value is None else str(value).strip()

    def molecule_from_smiles(self, value: Any) -> Any | None:
        if not self.smoke_tests["smiles_parse"]:
            return None
        smiles = self.normalize_smiles(value)
        if not smiles:
            return None
        if smiles not in self._molecule_cache:
            try:
                self._molecule_cache[smiles] = self._chem.MolFromSmiles(smiles)
            except Exception:
                self._molecule_cache[smiles] = None
        return self._molecule_cache[smiles]

    def smarts_matches(self, values: Iterable[Any], pattern: str) -> list[bool]:
        if not self.capabilities["smarts_filter"]:
            raise ChemistryError("SMARTS filtering requires optional RDKit support")
        try:
            query = self._chem.MolFromSmarts(str(pattern))
        except Exception as exc:
            raise ChemistryError("Invalid SMARTS pattern") from exc
        if query is None:
            raise ChemistryError("Invalid SMARTS pattern")

        matches: dict[str, bool] = {}
        result: list[bool] = []
        for value in values:
            smiles = self.normalize_smiles(value)
            if smiles not in matches:
                molecule = self.molecule_from_smiles(smiles)
                if molecule is None:
                    matches[smiles] = False
                else:
                    try:
                        matches[smiles] = bool(molecule.HasSubstructMatch(query))
                    except Exception:
                        matches[smiles] = False
            result.append(matches[smiles])
        return result

    def render_svg(self, value: Any, width: int = 560, height: int = 400) -> tuple[str, str]:
        if not self.capabilities["structure_render"]:
            raise ChemistryError("Structure rendering requires optional RDKit support")
        smiles = self.normalize_smiles(value)
        molecule = self.molecule_from_smiles(smiles)
        if molecule is None:
            raise ChemistryError("Invalid SMILES")
        try:
            drawer = self._drawer.MolDraw2DSVG(width, height)
            drawer.DrawMolecule(molecule)
            drawer.FinishDrawing()
            return smiles, drawer.GetDrawingText()
        except Exception as exc:
            raise ChemistryError("Could not render structure") from exc


SMILES_COLUMN_PRIORITY = ("smiles", "canonical_smiles", "isomeric_smiles")


def detect_smiles_column(columns: Iterable[Any], preferred: str = "SMILES") -> str | None:
    """Return the preferred column, then a known SMILES column, case-insensitively."""
    by_casefold = {str(column).casefold(): str(column) for column in columns}
    preferred_name = str(preferred).strip().casefold()
    if preferred_name and preferred_name in by_casefold:
        return by_casefold[preferred_name]
    return next((by_casefold[name] for name in SMILES_COLUMN_PRIORITY if name in by_casefold), None)
