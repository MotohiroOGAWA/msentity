Similarity results
==================

.. currentmodule:: msentity

:class:`SimilarityDataset` stores a pandas result table and the metadata used
to calculate it. Required table columns are ``index1``, ``index2``, and
``cosine_similarity``. Scores must be finite values from 0 through 1.

Calculate and save
------------------

Match spectra by unique, non-missing metadata keys and save the result as an
atomic ``.mssim`` file::

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
       bin_width=0.01,
       intensity_exponent=1.0,
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

API
---

.. autosummary::
   :toctree: generated/

   SimilarityDataset
