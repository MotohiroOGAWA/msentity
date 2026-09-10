Getting Started
===============

``msentity`` is a Python toolkit for reading, representing, editing, and
writing tandem mass-spectrometry datasets.  An :class:`msentity.MSDataset`
keeps spectrum metadata in a :class:`pandas.DataFrame` and stores all peak
lists compactly in a :class:`msentity.PeakSeries`.

It is designed for workflows that need:

- MSP, MGF, TSV, CSV, and native MSDS (HDF5) input
- spectrum- and peak-level metadata
- zero-copy dataset views for slicing, filtering, and sorting
- normalization, ID assignment, metadata joins, concatenation, similarity
  calculation, and export
- a command-line interface, interactive shell, and VS Code spectrum viewer

Requirements
------------

- Python 3.10 or later
- NumPy, pandas, h5py, PyArrow, and tqdm (installed automatically)
- For documentation builds: Sphinx, Furo, MyST Parser, and MyST-NB
- For the VS Code viewer: VS Code 1.90 or later and Node.js when building locally

Installation
------------

Install the current release directly from GitHub:

.. code-block:: bash

   python -m pip install "msentity @ git+https://github.com/MotohiroOGAWA/msentity.git"

For development, clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/MotohiroOGAWA/msentity.git
   cd msentity
   python -m pip install -e ".[docs]"

The VS Code viewer is installed separately from a release VSIX. See
:doc:`vscode_viewer` for download, installation, and build instructions.
See :doc:`cli` for command-line workflows.

Testing
-------

Run the complete test suite from the repository root:

.. code-block:: bash

   python -m unittest discover -s tests -p "Test*.py" -v

Build the documentation and treat warnings as errors:

.. code-block:: bash

   python -m sphinx -W -b html docs docs/_build/html

Basic Usage
-----------

Python examples in this guide are Jupyter cells.  The following cell creates a
small, self-contained dataset, so it can be copied directly into a notebook:

.. jupyter-input::

   import numpy as np
   import pandas as pd
   from msentity import MSDataset, PeakSeries

   metadata = pd.DataFrame({
       "Name": ["caffeine", "glucose"],
       "PrecursorMZ": [195.0877, 179.0561],
       "IonMode": ["Positive", "Negative"],
   })
   peaks = PeakSeries(
       data=np.array([
           [138.0662, 42.0], [195.0877, 100.0],
           [89.0244, 64.0], [179.0561, 100.0],
       ]),
       offsets=np.array([0, 2, 4], dtype=np.int64),
   )
   dataset = MSDataset(metadata, peaks, description="Notebook example")
   dataset

Integer indexing returns a :class:`msentity.SpectrumRecord`; slices and
boolean masks return lightweight dataset views:

.. jupyter-input::

   record = dataset[0]
   print(record["Name"], record.n_peaks)
   print(record.spectrum.mz)

   selected = dataset[dataset["PrecursorMZ"] > 180]
   selected.metadata
