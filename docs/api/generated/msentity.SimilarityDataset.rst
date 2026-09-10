msentity.SimilarityDataset
==========================

.. currentmodule:: msentity

.. autoclass:: SimilarityDataset
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   



Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :attr:`~msentity.SimilarityDataset.columns`
     -

   * - :attr:`~msentity.SimilarityDataset.has_matched_data`
     - Whether unique matched spectra and metadata are embedded.

   * - :attr:`~msentity.SimilarityDataset.matched_datasets`
     -

   * - :attr:`~msentity.SimilarityDataset.matched_source_indices`
     -

   * - :attr:`~msentity.SimilarityDataset.n_pairs`
     -

   * - :attr:`~msentity.SimilarityDataset.table`
     -

   * - :attr:`~msentity.SimilarityDataset.metadata`
     -




Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :meth:`~msentity.SimilarityDataset.__init__`
     - Initialize self.  See help(type(self)) for accurate signature.

   * - :meth:`~msentity.SimilarityDataset.copy`
     - Return an independent copy of the result table and metadata.

   * - :meth:`~msentity.SimilarityDataset.describe_scores`
     - Return count and five-number/central-tendency score statistics.

   * - :meth:`~msentity.SimilarityDataset.export_table`
     - Export the result table as Parquet, CSV, or TSV.

   * - :meth:`~msentity.SimilarityDataset.filter`
     - Return a result containing rows selected by a boolean mask.

   * - :meth:`~msentity.SimilarityDataset.from_datasets`
     - Calculate a similarity dataset from two mass-spectrum datasets.

   * - :meth:`~msentity.SimilarityDataset.histogram`
     - Return score frequencies and bin edges over the fixed range 0–1.

   * - :meth:`~msentity.SimilarityDataset.load`
     -

   * - :meth:`~msentity.SimilarityDataset.match_records`
     - Return query and reference records embedded for a result row.

   * - :meth:`~msentity.SimilarityDataset.save`
     - Atomically write a .mssim file; preserve an existing file on failure.

   * - :meth:`~msentity.SimilarityDataset.sort_values`
     - Return a stably sorted similarity result.

   * - :meth:`~msentity.SimilarityDataset.with_matched_data`
     - Attach each uniquely matched input spectrum once and index it from the table.




Property Details
----------------


.. autoattribute:: SimilarityDataset.columns


.. autoattribute:: SimilarityDataset.has_matched_data


.. autoattribute:: SimilarityDataset.matched_datasets


.. autoattribute:: SimilarityDataset.matched_source_indices


.. autoattribute:: SimilarityDataset.n_pairs


.. autoattribute:: SimilarityDataset.table


.. autoattribute:: SimilarityDataset.metadata





Method Details
--------------


.. automethod:: SimilarityDataset.__init__


.. automethod:: SimilarityDataset.copy


.. automethod:: SimilarityDataset.describe_scores


.. automethod:: SimilarityDataset.export_table


.. automethod:: SimilarityDataset.filter


.. automethod:: SimilarityDataset.from_datasets


.. automethod:: SimilarityDataset.histogram


.. automethod:: SimilarityDataset.load


.. automethod:: SimilarityDataset.match_records


.. automethod:: SimilarityDataset.save


.. automethod:: SimilarityDataset.sort_values


.. automethod:: SimilarityDataset.with_matched_data

