msentity.core.PeakSeries.PeakSeries
===================================

.. currentmodule:: msentity.core.PeakSeries

.. autoclass:: PeakSeries




Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :attr:`~msentity.core.PeakSeries.PeakSeries.count`
     - Number of spectra in the current view.

   * - :attr:`~msentity.core.PeakSeries.PeakSeries.data`
     - Peak array of the current view.

   * - :attr:`~msentity.core.PeakSeries.PeakSeries.intensity`
     - Intensity values of the current view.

   * - :attr:`~msentity.core.PeakSeries.PeakSeries.lengths`
     - Number of peaks in each visible spectrum.

   * - :attr:`~msentity.core.PeakSeries.PeakSeries.metadata`
     - Peak-level metadata of the current view.

   * - :attr:`~msentity.core.PeakSeries.PeakSeries.metadata_columns`
     - Metadata columns exposed by the current view.

   * - :attr:`~msentity.core.PeakSeries.PeakSeries.mz`
     - m/z values of the current view.

   * - :attr:`~msentity.core.PeakSeries.PeakSeries.n_peaks_total`
     - Total number of peaks in the current view.

   * - :attr:`~msentity.core.PeakSeries.PeakSeries.offsets`
     - Offsets of the current view.

   * - :attr:`~msentity.core.PeakSeries.PeakSeries.peak_indices`
     - Global peak indices corresponding to the current view.




Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :meth:`~msentity.core.PeakSeries.PeakSeries.__init__`
     - Initialize self.  See help(type(self)) for accurate signature.

   * - :meth:`~msentity.core.PeakSeries.PeakSeries.copy`
     - Materialize the current view as an independent object.

   * - :meth:`~msentity.core.PeakSeries.PeakSeries.n_peaks`
     - Number of peaks in a specific visible spectrum.

   * - :meth:`~msentity.core.PeakSeries.PeakSeries.normalize`
     - Normalize intensities so that the maximum intensity of each spectrum becomes ``scale``.

   * - :meth:`~msentity.core.PeakSeries.PeakSeries.reorder`
     - Reorder spectra in the current view.

   * - :meth:`~msentity.core.PeakSeries.PeakSeries.sort_by_intensity`
     - Sort peaks by intensity within each spectrum.

   * - :meth:`~msentity.core.PeakSeries.PeakSeries.sort_by_mz`
     - Sort peaks by m/z within each spectrum.




Property Details
----------------


.. autoattribute:: PeakSeries.count


.. autoattribute:: PeakSeries.data


.. autoattribute:: PeakSeries.intensity


.. autoattribute:: PeakSeries.lengths


.. autoattribute:: PeakSeries.metadata


.. autoattribute:: PeakSeries.metadata_columns


.. autoattribute:: PeakSeries.mz


.. autoattribute:: PeakSeries.n_peaks_total


.. autoattribute:: PeakSeries.offsets


.. autoattribute:: PeakSeries.peak_indices





Method Details
--------------


.. automethod:: PeakSeries.__init__


.. automethod:: PeakSeries.copy


.. automethod:: PeakSeries.n_peaks


.. automethod:: PeakSeries.normalize


.. automethod:: PeakSeries.reorder


.. automethod:: PeakSeries.sort_by_intensity


.. automethod:: PeakSeries.sort_by_mz


