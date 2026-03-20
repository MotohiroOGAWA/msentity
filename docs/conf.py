import os
import sys

import importlib
import inspect

sys.path.insert(0, os.path.abspath(".."))

print("CONF:", __file__)
print("PATH ADDED:", os.path.abspath(".."))
print("SYS.PATH[0]:", sys.path[0])

try:
    import msentity
    print("msentity imported:", msentity)
    print("msentity file:", getattr(msentity, "__file__", None))
except Exception as e:
    print("FAILED import msentity:", repr(e))
    raise


def get_summary(module_name: str, obj_name: str, member_name: str) -> str:
    candidates = [module_name]
    if "." in module_name:
        candidates.append(module_name.rsplit(".", 1)[0])

    for mod_name in candidates:
        try:
            module = importlib.import_module(mod_name)
            obj = getattr(module, obj_name)
            member = getattr(obj, member_name)

            if isinstance(member, property):
                doc = inspect.getdoc(member.fget)
            else:
                doc = inspect.getdoc(member)

            if not doc:
                continue

            for line in doc.splitlines():
                line = line.strip()
                if line:
                    return line
        except Exception:
            continue

    return ""


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'msentity'
copyright = '2026, MotohiroOGAWA'
author = 'MotohiroOGAWA'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autosummary_context = {
    "get_summary": get_summary,
}

autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']
