Inspecting datasets
===================

This page explains how to inspect dataset-level information and spectrum
metadata.

Loading an example dataset
--------------------------

.. code-block:: python

   from msentity import load_ms_dataset

   dataset = load_ms_dataset("example.msp")

Dataset summary
---------------

The command line interface provides ``msentity info``. In Python, the same
information can be obtained from the dataset object.

.. code-block:: python

   summary = {
       "n_spectra": len(dataset),
       "n_columns": dataset.n_columns,
       "n_peaks_total": dataset.n_peaks_total,
       "columns": dataset.columns,
       "description": dataset.description,
       "attributes": dataset.attributes,
       "tags": dataset.tags,
   }

   summary

Example output:

.. code-block:: python

   {
       "n_spectra": 3,
       "n_columns": 4,
       "n_peaks_total": 12,
       "columns": ["Name", "PrecursorMZ", "IonMode", "AdductType"],
       "description": "",
       "attributes": {},
       "tags": [],
   }

Previewing metadata
-------------------

The CLI command:

.. code-block:: bash

   msentity head example.msp -n 10

corresponds to:

.. code-block:: python

   dataset.metadata.head(10)

Example output:

.. code-block:: text

             Name  PrecursorMZ   IonMode AdductType
   0   Compound_A     301.2162  POSITIVE     [M+H]+
   1   Compound_B     255.1234  NEGATIVE     [M-H]-
   2   Compound_C     412.2871  POSITIVE    [M+Na]+

Accessing metadata columns
--------------------------

.. code-block:: python

   dataset["Name"]

.. code-block:: python

   dataset["PrecursorMZ"]

Selecting visible columns
-------------------------

The visible metadata columns are stored in ``dataset.columns``.

.. code-block:: python

   dataset.columns

You can change the visible columns.

.. code-block:: python

   dataset.columns = [
       "Name",
       "PrecursorMZ",
       "IonMode",
   ]

   dataset.metadata.head()