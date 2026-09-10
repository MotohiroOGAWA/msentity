Similarity results
==================

.. currentmodule:: msentity

:class:`SimilarityDataset` stores a pandas result table, calculation metadata,
and optional matched spectra. Required table columns are ``index1``,
``index2``, and ``cosine_similarity``. The score column keeps this name for
file compatibility when reverse cosine is selected. Scores are finite values
from 0 through 1.

Library search
--------------

Library search compares every query spectrum with every reference spectrum.
The implementation bins peak arrays with NumPy, processes flattened candidate
pairs in bounded chunks, and retains only scores at or above ``threshold``.
It therefore does not keep the full all-pairs matrix in memory::

   from msentity import calculate_library_search, load_ms_dataset

   query = load_ms_dataset("query.msds")
   reference = load_ms_dataset("reference.msp")
   result = calculate_library_search(
       query,
       reference,
       query_source="query.msds",
       reference_source="reference.msp",
       method="cosine",       # or "reverse_cosine"
       threshold=0.8,
       bin_width=0.01,
   )
   result.save("library-search.mssim")

The library-search default embeds matched data. Each unique matched query and
reference record is stored once; ``data_index1`` and ``data_index2`` in the
result table refer to those compact datasets. Original zero-based input
positions remain in ``index1`` and ``index2``. This avoids repeating pandas
metadata and NumPy peak arrays when one spectrum has many matches. Access an
embedded pair with ``result.match_records(row)``.

Set ``include_matched_data=False`` for a lightweight file containing the match
table and calculation/source metadata only. Saving a filtered embedded result
automatically removes matched records no longer referenced by its table.

Calculate and save
------------------

To compare only records with a shared ID, match spectra by unique, non-missing
metadata keys and save an atomic ``.mssim`` file::

   from msentity import SimilarityDataset, load_ms_dataset

   first = load_ms_dataset("first.msds")
   second = load_ms_dataset("second.msds")
   result = SimilarityDataset.from_datasets(
       first,
       second,
       source1="first.msds",
       source2="second.msds",
       key1="SpecID",
       key2="SpecID",
       method="cosine",
       bin_width=0.01,
       intensity_exponent=1.0,
       include_matched_data=False,
   )
   result.save("result.mssim")

Duplicate non-missing keys are rejected because pairing would be ambiguous.
Missing and unmatched keys are skipped, and output follows the first dataset's
row order. ``index1`` and ``index2`` are zero-based input positions.

Inspect and export
------------------

Load a result, select rows, calculate summaries, and export the table::

   loaded = SimilarityDataset.load("result.mssim")
   selected = loaded.filter(loaded.table["cosine_similarity"] >= 0.8)
   selected = selected.sort_values("cosine_similarity", ascending=False)
   statistics = selected.describe_scores()
   counts, edges = selected.histogram(bins=20)
   selected.export_table("selected.csv")

``export_table`` accepts ``.csv``, ``.tsv``, and ``.parquet``. These table
formats omit calculation metadata; use ``save`` with ``.mssim`` to retain it.

MSSIM storage
-------------

Schema version 2 is an HDF5 container with a Parquet result table and JSON
calculation metadata. Embedded results also have two ``matched_data`` groups.
Each group stores the compact matched dataset, including its pandas metadata
and NumPy-backed peaks, plus the corresponding original source indices.
Version 1 table-only files remain readable.

API
---

- :class:`msentity.SimilarityDataset`
- :func:`msentity.similarity.calculate_library_search`
- :func:`msentity.similarity.calculate_similarity`
- :func:`msentity.similarity.library_search`
- :func:`msentity.similarity.similarity_by_key`

.. toctree::
   :hidden:

   generated/msentity.SimilarityDataset
