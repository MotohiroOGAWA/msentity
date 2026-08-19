Inspecting datasets
===================

This page explains how to inspect a dataset without reaching into private
attributes.

Loading an example dataset
--------------------------

.. jupyter-input::

   from msentity import load_ms_dataset
   dataset = load_ms_dataset("example.msp")

Dataset summary
---------------

.. jupyter-input::

   summary = {
       "n_spectra": len(dataset),
       "shape": dataset.shape,
       "n_columns": dataset.n_columns,
       "n_peaks_total": dataset.n_peaks_total,
       "columns": dataset.columns,
       "description": dataset.description,
       "attributes": dataset.attributes,
       "tags": dataset.tags,
   }
   summary

``n_rows`` is an alias-like property for ``len(dataset)``. ``repr(dataset)``
also gives a concise spectrum, peak, and visible-column summary.

Previewing metadata
-------------------

The CLI equivalent is:

.. code-block:: bash

   msentity head example.msp --num-rows 10

In a notebook, use normal pandas operations:

.. jupyter-input::

   dataset.metadata.head(10)

The returned table is reset to a zero-based index and should be treated as a
read-only snapshot. Assign through ``dataset[column]`` or a record to make a
reliable change.

Accessing metadata columns
--------------------------

.. jupyter-input::

   names = dataset["Name"]
   precursor_mz = dataset["PrecursorMZ"]
   names.head(), precursor_mz.describe()

Only visible columns can be accessed this way; a missing or hidden column
raises :class:`KeyError`.

Selecting visible columns
-------------------------

``columns`` changes presentation, not the underlying stored metadata:

.. jupyter-input::

   dataset.columns = ["Name", "PrecursorMZ", "IonMode"]
   dataset.metadata.head()

Use ``dataset.reset_view(reset_columns=True)`` to expose every underlying
column again. Use ``dataset.copy()`` when an independent, materialized dataset
is needed instead of a view that shares data.
