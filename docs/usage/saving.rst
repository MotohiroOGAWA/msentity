Saving datasets
===============

MSDS is the lossless native format. MSP and MGF are interoperability formats,
so only fields representable by those formats are exported.

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

   filtered = dataset[dataset["PrecursorMZ"] > 300]
   filtered.save("filtered.msds", save_view=True)

Use ``save_view=False`` only when intentionally saving the complete underlying
references rather than the current view.

Saving MSP and MGF files
------------------------

.. jupyter-input::

   from msentity import write_mgf, write_msp

   write_msp(dataset, "output.msp")
   write_mgf(dataset, "output.mgf")

The writer call order is ``(dataset, output_path)``. For conversion on the
command line, ``msentity convert`` selects the output from its extension.
``msentity merge-dir`` can recursively combine a directory of MSP/MGF/MSDS
files and optionally attach source paths and source spectrum indices.
