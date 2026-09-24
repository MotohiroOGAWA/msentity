Getting Started
===============

``msentity`` reads, edits, compares, and writes tandem mass-spectrometry
datasets.  :class:`msentity.MSDataset` keeps one metadata row and one peak
list per spectrum. This page uses a complete three-spectrum MSP file so that
every result can be reproduced.

Requirements
------------

* Python 3.10 or later
* NumPy, pandas, h5py, PyArrow, and tqdm (installed with ``msentity``)
* VS Code 1.90 or later only when using the Spectrum Viewer

Installation and verification
-----------------------------

Install the current release from GitHub, then print the installed version and
the command-line help:

.. code-block:: console

   $ python -m pip install "msentity @ git+https://github.com/MotohiroOGAWA/msentity.git"
   $ python -c "import importlib.metadata; print(importlib.metadata.version('msentity'))"
   0.1.0
   $ msentity --help
   usage: msentity [-h]
                   {convert,head,info,library-search,merge-dir,meta,similarity-by-key,shell}
                   ...

The version is an example; a newer release may print a larger number. For a
development checkout:

.. code-block:: console

   $ git clone https://github.com/MotohiroOGAWA/msentity.git
   $ cd msentity
   $ python -m pip install -e ".[docs]"

The example input
-----------------

Save the following as ``example.msp``. It is also included in the repository
at ``docs/examples/example.msp``.

.. literalinclude:: examples/example.msp
   :language: text

Load and inspect the result
---------------------------

Run this from the directory containing ``example.msp``:

.. code-block:: python

   from msentity import load_ms_dataset

   dataset = load_ms_dataset("example.msp")
   print(dataset)
   print(dataset.metadata.to_string(index=False))

Output (the reader may also display a progress bar):

.. code-block:: text

   MSDataset(n_spectra=3, n_peaks=12, columns=['Name', 'PrecursorMZ', 'AdductType', 'CollisionEnergy', 'NumPeaks'])
         Name PrecursorMZ AdductType CollisionEnergy NumPeaks
   Compound_A    301.2162     [M+H]+              20        4
   Compound_B    255.1234     [M+H]+              30        3
   Compound_C    412.2871    [M+Na]+              25        5

MSP field names are canonicalized: for example, ``Precursor_type`` becomes
``AdductType`` and ``Collision_energy`` becomes ``CollisionEnergy``. MSP
metadata values remain strings. Peak intensities are normalized to a maximum
of 1.0 for each spectrum when read.

Inspect one spectrum
--------------------

.. code-block:: python

   import pandas as pd

   record = dataset[0]
   print(record["Name"], record.n_peaks)
   peaks = pd.DataFrame(record.spectrum.data, columns=["mz", "intensity"])
   print(peaks.to_string(index=False))

Output:

.. code-block:: text

   Compound_A 4
      mz  intensity
   100.0      0.120
   145.1      0.553
   183.2      0.217
   301.2      1.000

Integer indexing returns a :class:`msentity.SpectrumRecord`. Slices and
boolean masks return lightweight dataset views:

.. code-block:: python

   precursor_mz = dataset["PrecursorMZ"].astype(float)
   selected = dataset[precursor_mz > 300]
   print(selected.metadata[["Name", "PrecursorMZ"]].to_string(index=False))

.. code-block:: text

         Name PrecursorMZ
   Compound_A    301.2162
   Compound_C    412.2871

Next steps
----------

* :doc:`cli` covers commands and the interactive shell with transcripts.
* :doc:`usage/index` develops the same example through the Python API.
* :doc:`vscode_viewer` explains the graphical table, plots, comparison, and
  export workflows.

Development checks
------------------

.. code-block:: console

   $ python -m unittest discover -s tests -p "Test*.py" -v
   $ python -m sphinx -W -b html docs docs/_build/html
