Saving datasets
===============

MSDS is the lossless native format. MSP, MGF, TSV, and CSV are interchange
formats, so dataset-level metadata and peak annotations that those formats
cannot represent are not preserved.

Saving as MSDS
--------------

The CLI command is:

.. code-block:: bash

   msentity convert example.msp example.msds

The Python equivalent is:

.. jupyter-input::

   from msentity import load_ms_dataset
   dataset = load_ms_dataset("example.msp")
   dataset.save("example.msds")

``mode="w"`` (the default) creates/replaces the file; ``mode="a"`` passes
append mode to the MSDS writer.

Loading the saved file
----------------------

.. jupyter-input::

   from msentity import MSDataset

   loaded = load_ms_dataset("example.msds")
   loaded_directly = MSDataset.load("example.msds")
   loaded

Pass ``load_peak_metadata=False`` to ``MSDataset.load`` when peak annotations
are not needed and memory use matters.

Saving only the current view
----------------------------

``save_view=True`` is the default, so filtering, ordering, visible columns,
and corresponding peaks are materialized consistently:

.. jupyter-input::

   import pandas as pd
   precursor_mz = pd.to_numeric(dataset["PrecursorMZ"], errors="coerce")
   filtered = dataset[precursor_mz > 300]
   filtered.save("filtered.msds", save_view=True)

Use ``save_view=False`` only when intentionally saving the complete underlying
references rather than the current view.

Saving MSP and MGF files
------------------------

.. jupyter-input::

   from msentity import write_mgf, write_msp

   write_msp(dataset, "output.msp")
   write_mgf(dataset, "output.mgf")

Saving TSV and CSV files
------------------------

Both formats write one spectrum per row. The final ``Peak`` column uses
``mz1,intensity1;mz2,intensity2;...``::

   from msentity import write_csv, write_tsv

   write_tsv(dataset, "output.tsv")
   write_csv(dataset, "output.csv")
   write_csv(dataset, "selected.csv", headers=["SpecID", "Name"])

CSV fields containing commas, quotes, or newlines are quoted automatically.
The writer call order is ``(dataset, output_path)``; ``headers`` selects and
orders metadata columns. Read the results with ``load_ms_dataset``,
``read_tsv``, or ``read_csv``.

The ``msentity convert`` command currently writes MSDS only.
``msentity merge-dir`` can recursively combine MSP, MGF, MSDS, TSV, and CSV
inputs into an MSDS file and can attach source paths and spectrum indices.
