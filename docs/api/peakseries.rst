PeakSeries
==========

.. currentmodule:: msentity.core.PeakSeries

Represents peak-level data across multiple spectra.

The :class:`~msentity.core.PeakSeries.PeakSeries` class stores all peaks in a compact
array-based structure and uses offsets to separate individual spectra.

This design enables efficient vectorized operations and scalable handling
of large mass spectrometry datasets.

Class
-----

Core data structure for peak storage and manipulation.

.. autosummary::
   :toctree: generated/

   PeakSeries