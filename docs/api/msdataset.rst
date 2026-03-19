MSDataset
=========

.. currentmodule:: msentity

Represents a dataset of mass spectra and associated metadata.

The :class:`~msentity.MSDataset` class combines spectrum-level metadata
stored in a pandas DataFrame with peak-level data stored in a
:class:`~msentity.core.PeakSeries.PeakSeries`.

Each row in the metadata corresponds to one spectrum in the peak series.
The class provides functionality for slicing, filtering, merging metadata,
sorting, and performing efficient I/O operations such as HDF5 serialization.

Class
-----

Core class for managing large-scale mass spectrometry datasets.

.. autosummary::
   :toctree: generated/

   MSDataset