msentity.SpectrumRecord
=======================

.. currentmodule:: msentity

.. autoclass:: SpectrumRecord
   :members:
   :undoc-members:
   :show-inheritance:
   



Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :attr:`~msentity.SpectrumRecord.columns`
     - Metadata columns available in the current view.

   * - :attr:`~msentity.SpectrumRecord.is_integer_mz`
     - Check whether all m/z values are integers.

   * - :attr:`~msentity.SpectrumRecord.n_peaks`
     - Number of peaks in the spectrum.

   * - :attr:`~msentity.SpectrumRecord.peaks`
     - Alias for :attr:`spectrum`.

   * - :attr:`~msentity.SpectrumRecord.spectrum`
     - Peak data of the record.




Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :meth:`~msentity.SpectrumRecord.__init__`
     - Initialize self.  See help(type(self)) for accurate signature.

   * - :meth:`~msentity.SpectrumRecord.copy`
     - Materialize this record as an independent object.

   * - :meth:`~msentity.SpectrumRecord.normalize`
     - Normalize the intensities of the spectrum.

   * - :meth:`~msentity.SpectrumRecord.sort_by_intensity`
     - Sort peaks by intensity.

   * - :meth:`~msentity.SpectrumRecord.sort_by_mz`
     - Sort peaks by m/z.




Property Details
----------------


.. autoattribute:: SpectrumRecord.columns


.. autoattribute:: SpectrumRecord.is_integer_mz


.. autoattribute:: SpectrumRecord.n_peaks


.. autoattribute:: SpectrumRecord.peaks


.. autoattribute:: SpectrumRecord.spectrum





Method Details
--------------


.. automethod:: SpectrumRecord.__init__


.. automethod:: SpectrumRecord.copy


.. automethod:: SpectrumRecord.normalize


.. automethod:: SpectrumRecord.sort_by_intensity


.. automethod:: SpectrumRecord.sort_by_mz


