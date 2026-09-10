Development
===========

This section describes how to work on ``msentity`` locally.

Local setup and checks
----------------------

Clone the repository and install the package and documentation dependencies::

   git clone https://github.com/MotohiroOGAWA/msentity.git
   cd msentity
   python -m pip install -e ".[docs]"

Run the Python tests and build the documentation with warnings treated as
errors::

   python -m unittest discover -s tests -p "Test*.py" -v
   python -m unittest discover -s vscode-extension/tests -p "test_*.py" -v
   python -m sphinx -W -b html docs docs/_build/html

The browser interaction test additionally needs Playwright and Chromium. The
VS Code extension build and release procedure is documented in
:doc:`vscode_viewer`.

Using as a Git Submodule
------------------------

You can integrate ``msentity`` into an existing project as a Git submodule.

.. code-block:: bash

   # At the root directory of your project
   git submodule add https://github.com/MotohiroOGAWA/msentity.git ./msentity
   git commit -m "Add msentity as submodule"

Notes
-----

Using a Git submodule is useful for:

- Reproducible research environments
- Managing dependencies in HPC or cluster environments
- Integrating ``msentity`` into larger pipelines
