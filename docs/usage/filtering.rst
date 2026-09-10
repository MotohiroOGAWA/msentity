Filtering datasets
==================

Filtering uses pandas boolean expressions and returns an
:class:`msentity.MSDataset` view. Spectrum metadata and peak lists always stay
aligned.

Loading a dataset
-----------------

.. jupyter-input::

   import pandas as pd
   from msentity import load_ms_dataset
   dataset = load_ms_dataset("example.msp")

Filtering by numeric metadata
-----------------------------

The interactive-shell expression ``filter PrecursorMZ > 300`` corresponds to:

.. jupyter-input::

   precursor_mz = pd.to_numeric(dataset["PrecursorMZ"], errors="coerce")
   filtered = dataset[precursor_mz > 300]
   filtered.metadata

Text formats retain metadata values as strings unless the format-specific
reader can infer a numeric column. ``pd.to_numeric(..., errors="coerce")``
makes the comparison explicit and treats non-numeric values as missing.

Filtering by ion mode
---------------------

Values depend on the source file; inspect unique values before comparing:

.. jupyter-input::

   dataset["IonMode"].dropna().unique()
   positive = dataset[
       dataset["IonMode"].astype(str).str.casefold() == "positive"
   ]

Filtering by adduct
-------------------

MSP ``PRECURSORTYPE`` is canonicalized to ``AdductType``:

.. jupyter-input::

   protonated = dataset[dataset["AdductType"] == "[M+H]+"]
   protonated.metadata

Use ``dataset.columns`` first because source formats are not required to carry
this optional field.

Filtering by text
-----------------

The shell expression ``filter Name contains glucose`` corresponds to:

.. jupyter-input::

   glucose_like = dataset[
       dataset["Name"].astype(str).str.contains(
           "glucose", case=False, na=False, regex=False
       )
   ]
   glucose_like.metadata

``regex=False`` treats punctuation in the query literally.

Combining multiple filters
--------------------------

Parenthesize each pandas condition and combine them with ``&``, ``|``, and
``~``:

.. jupyter-input::

   selected = dataset[
       (dataset["IonMode"].astype(str).str.casefold() == "positive")
       & precursor_mz.between(200, 500)
   ]
   selected = selected.sort_by("PrecursorMZ", ascending=False)
   selected.metadata

Integer lists and slices are also supported: ``dataset[[0, 3, 5]]`` and
``dataset[:10]``.

Resetting views
---------------

Views share their underlying arrays with the source dataset. ``reset_view``
restores all source spectra; it does not undo edits:

.. jupyter-input::

   full_view = filtered.reset_view(in_place=False)
   len(filtered), len(full_view)

The default ``in_place=True`` modifies the view object. To detach filtered
data and peaks completely, use ``materialized = filtered.copy()``.
