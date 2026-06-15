Saving datasets
===============

This page explains how to save datasets.

Saving as MSDS
--------------

The CLI command:

.. code-block:: bash

   msentity convert example.msp example.msds

corresponds to:

.. code-block:: python

   from msentity import load_ms_dataset

   dataset = load_ms_dataset("example.msp")

   dataset.save("example.msds")

Loading the saved file
----------------------

.. code-block:: python

   loaded = load_ms_dataset("example.msds")

   loaded

You can also load an MSDS file with :meth:`msentity.MSDataset.load`.

.. code-block:: python

   from msentity import MSDataset

   loaded = MSDataset.load("example.msds")

Saving only the current view
----------------------------

By default, :meth:`save` saves the current view.

.. code-block:: python

   filtered = dataset[dataset["PrecursorMZ"] > 300]

   filtered.save("filtered.msds")

Saving MSP and MGF files
------------------------

Use :func:`write_msp` and :func:`write_mgf` to export datasets.

.. code-block:: python

   from msentity import write_msp, write_mgf

   write_msp(dataset, "output.msp")
   write_mgf(dataset, "output.mgf")