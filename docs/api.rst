API Reference
=============

The ``msentity`` API provides efficient data structures for handling
mass spectrometry data, including spectrum-level metadata and peak-level data.

This reference documents the core modules of the package:

- ``MSDataset``: dataset abstraction combining metadata and peak data
- ``SpectrumRecord``: access to a single spectrum with associated metadata
- ``PeakSeries``: storage and manipulation of peak-level data across multiple spectra
- ``Spectrum``: view of a single spectrum with m/z–intensity pairs and operations

These classes are designed to support mass spectrometry workflows such as
data processing, filtering, sorting, normalization, and dataset-level operations.

Classes
=======

.. list-table::
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`MSDataset <api/generated/msentity.MSDataset>`
     - Represents a dataset combining spectrum metadata and peak data with support for slicing and I/O.
   * - :doc:`SpectrumRecord <api/generated/msentity.SpectrumRecord>`
     - Represents a single spectrum entry with convenient access to metadata and peak data.
   * - :doc:`PeakSeries <api/generated/msentity.core.PeakSeries.PeakSeries>`
     - Represents peak-level data across multiple spectra using a compact array-based structure.
   * - :doc:`Spectrum <api/generated/msentity.Spectrum>`
     - Represents a single spectrum with m/z and intensity arrays and basic operations.


.. toctree::
   :maxdepth: 2
   :hidden:

   api/msdataset
   api/spectrumrecord
   api/peakseries
   api/spectrum