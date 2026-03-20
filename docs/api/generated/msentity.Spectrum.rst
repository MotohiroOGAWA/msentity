msentity.Spectrum
=================

.. currentmodule:: msentity

.. autoclass:: Spectrum
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

   * - :attr:`~msentity.Spectrum.data`
     - Peak array of the spectrum.

   * - :attr:`~msentity.Spectrum.intensity`
     - Intensity values of the spectrum.

   * - :attr:`~msentity.Spectrum.metadata`
     - Peak-level metadata of the spectrum.

   * - :attr:`~msentity.Spectrum.mz`
     - m/z values of the spectrum.




Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description

   * - :meth:`~msentity.Spectrum.__init__`
     - Initialize self.  See help(type(self)) for accurate signature.

   * - :meth:`~msentity.Spectrum.normalize`
     - Normalize intensities so that the maximum intensity becomes ``scale``.

   * - :meth:`~msentity.Spectrum.sort_by_intensity`
     - Sort peaks by intensity.

   * - :meth:`~msentity.Spectrum.sort_by_mz`
     - Sort peaks by m/z.




Property Details
----------------


.. autoattribute:: Spectrum.data


.. autoattribute:: Spectrum.intensity


.. autoattribute:: Spectrum.metadata


.. autoattribute:: Spectrum.mz





Method Details
--------------


.. automethod:: Spectrum.__init__


.. automethod:: Spectrum.normalize


.. automethod:: Spectrum.sort_by_intensity


.. automethod:: Spectrum.sort_by_mz


