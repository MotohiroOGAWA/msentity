PeakSeries
==========

.. currentmodule:: msentity.core.PeakSeries

Represents peak-level data across multiple spectra.

The :class:`~msentity.core.PeakSeries.PeakSeries` class stores all peaks in a compact
array-based structure and uses offsets to separate individual spectra.

This design enables efficient vectorized operations and scalable handling
of large mass spectrometry datasets.

In addition to ``mz`` and ``intensity`` arrays, a series can carry a pandas
DataFrame of peak annotations. Subsetting and peak sorting preserve alignment.
Normalization is performed independently per spectrum; most transformations
support an ``in_place`` option.

Class
-----

Core data structure for peak storage and manipulation.

.. autosummary::
   :toctree: generated/

   PeakSeries
