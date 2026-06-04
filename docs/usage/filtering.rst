Filtering datasets
==================

This page explains how to filter spectra by spectrum-level metadata.

Loading a dataset
-----------------

.. code-block:: python

   from msentity import load_ms_dataset

   dataset = load_ms_dataset("example.msp")

Filtering by numeric metadata
-----------------------------

The shell command:

.. code-block:: text

   filter PrecursorMZ > 300

corresponds to:

.. code-block:: python

   filtered = dataset[dataset["PrecursorMZ"] > 300]

   filtered.metadata

Filtering by ion mode
---------------------

.. code-block:: python

   positive = dataset[dataset["IonMode"] == "POSITIVE"]

   positive.metadata

Filtering by adduct
-------------------

.. code-block:: python

   protonated = dataset[dataset["AdductType"] == "[M+H]+"]

   protonated.metadata

Filtering by text
-----------------

The shell command:

.. code-block:: text

   filter Name contains glucose

corresponds to:

.. code-block:: python

   glucose_like = dataset[
       dataset["Name"].astype(str).str.contains(
           "glucose",
           case=False,
           na=False,
       )
   ]

   glucose_like.metadata

Combining multiple filters
--------------------------

.. code-block:: python

   selected = dataset[
       (dataset["IonMode"] == "POSITIVE")
       & (dataset["PrecursorMZ"] > 300)
   ]

   selected.metadata

Resetting views
---------------

Filtering usually creates a new dataset object, so the original dataset remains
available.

.. code-block:: python

   len(dataset), len(filtered)

If you intentionally changed a dataset view and want to restore it, use
``reset_view``.

.. code-block:: python

   dataset.reset_view()