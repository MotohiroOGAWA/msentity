msentity.MSDataset
==================

.. currentmodule:: msentity

.. autoclass:: MSDataset
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

   * - :attr:`~msentity.MSDataset.attributes`
     - Dataset-level attributes.

   * - :attr:`~msentity.MSDataset.columns`
     - Spectrum metadata columns exposed by the current view.

   * - :attr:`~msentity.MSDataset.description`
     - Dataset description.

   * - :attr:`~msentity.MSDataset.metadata`
     - Spectrum-level metadata for the current view.

   * - :attr:`~msentity.MSDataset.n_columns`
     - Number of visible metadata columns.

   * - :attr:`~msentity.MSDataset.n_peaks_total`
     - Total number of peaks across all visible spectra.

   * - :attr:`~msentity.MSDataset.n_rows`
     - Number of spectra in the current view.

   * - :attr:`~msentity.MSDataset.peaks`
     - Peak series associated with the current view.

   * - :attr:`~msentity.MSDataset.shape`
     - Shape of the dataset view.

   * - :attr:`~msentity.MSDataset.tags`
     - Dataset tags.




Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :meth:`~msentity.MSDataset.__init__`
     - Initialize self.  See help(type(self)) for accurate signature.

   * - :meth:`~msentity.MSDataset.add_tag`
     - Add a tag if it does not already exist.

   * - :meth:`~msentity.MSDataset.clear_attributes`
     - Remove all dataset attributes.

   * - :meth:`~msentity.MSDataset.clear_tags`
     - Remove all dataset tags.

   * - :meth:`~msentity.MSDataset.concat`
     - Concatenate multiple datasets.

   * - :meth:`~msentity.MSDataset.copy`
     - Materialize the current view as an independent dataset.

   * - :meth:`~msentity.MSDataset.has_attribute`
     - Check whether an attribute exists.

   * - :meth:`~msentity.MSDataset.has_tag`
     - Check whether a tag exists.

   * - :meth:`~msentity.MSDataset.load`
     - Load a dataset from an .msds file.

   * - :meth:`~msentity.MSDataset.merge_metadata`
     - Merge an external DataFrame into the spectrum metadata of the current view.

   * - :meth:`~msentity.MSDataset.read_dataset_meta`
     - Read only dataset-level metadata from an HDF5 file.

   * - :meth:`~msentity.MSDataset.remove_attribute`
     - Remove a dataset attribute.

   * - :meth:`~msentity.MSDataset.remove_tag`
     - Remove a tag.

   * - :meth:`~msentity.MSDataset.reset_view`
     - Reset the current dataset view to the full dataset.

   * - :meth:`~msentity.MSDataset.save`
     - Save the dataset to an .msds file.

   * - :meth:`~msentity.MSDataset.set_attribute`
     - Add or update a dataset attribute.

   * - :meth:`~msentity.MSDataset.sort_by`
     - Sort spectra by a spectrum metadata column.




Property Details
----------------


.. autoattribute:: MSDataset.attributes


.. autoattribute:: MSDataset.columns


.. autoattribute:: MSDataset.description


.. autoattribute:: MSDataset.metadata


.. autoattribute:: MSDataset.n_columns


.. autoattribute:: MSDataset.n_peaks_total


.. autoattribute:: MSDataset.n_rows


.. autoattribute:: MSDataset.peaks


.. autoattribute:: MSDataset.shape


.. autoattribute:: MSDataset.tags





Method Details
--------------


.. automethod:: MSDataset.__init__


.. automethod:: MSDataset.add_tag


.. automethod:: MSDataset.clear_attributes


.. automethod:: MSDataset.clear_tags


.. automethod:: MSDataset.concat


.. automethod:: MSDataset.copy


.. automethod:: MSDataset.has_attribute


.. automethod:: MSDataset.has_tag


.. automethod:: MSDataset.load


.. automethod:: MSDataset.merge_metadata


.. automethod:: MSDataset.read_dataset_meta


.. automethod:: MSDataset.remove_attribute


.. automethod:: MSDataset.remove_tag


.. automethod:: MSDataset.reset_view


.. automethod:: MSDataset.save


.. automethod:: MSDataset.set_attribute


.. automethod:: MSDataset.sort_by


