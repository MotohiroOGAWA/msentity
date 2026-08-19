Records and peaks
=================

This page covers individual :class:`msentity.SpectrumRecord` objects,
:class:`msentity.Spectrum` views, and flattened :class:`msentity.PeakSeries`
storage.

Loading a dataset
-----------------

.. jupyter-input::

   import pandas as pd
   from msentity import load_ms_dataset
   dataset = load_ms_dataset("example.msp")

Accessing one spectrum record
-----------------------------

The interactive-shell command ``show 0`` corresponds to integer indexing:

.. jupyter-input::

   record = dataset[0]
   record

Unlike ``dataset.metadata.iloc[0]``, a record keeps its metadata and peak list
together. Iterating over a dataset yields records in the current view order.

Record-level metadata
---------------------

.. jupyter-input::

   record["Name"]
   record["PrecursorMZ"]
   record.columns

Assignment adds or updates a field in the parent dataset:

.. jupyter-input::

   record["Reviewed"] = True
   assert "Reviewed" in record

Useful properties include ``record.n_peaks``, ``record.is_integer_mz``,
``record.spectrum``, and its alias ``record.peaks``. ``record.copy()`` creates
an independent one-spectrum dataset record.

Accessing peaks
---------------

``dataset.peaks`` is a ``PeakSeries``; indexing it returns a ``Spectrum``:

.. jupyter-input::

   spectrum = dataset.peaks[0]       # same peak data as record.spectrum
   peaks = pd.DataFrame(spectrum.data, columns=["mz", "intensity"])
   peaks

Direct arrays are available as ``spectrum.mz`` and ``spectrum.intensity``.
Across the current dataset view, use ``dataset.peaks.data``, ``.mz``,
``.intensity``, ``.offsets``, ``.lengths``, and ``.n_peaks_total``. If the
source contains peak annotations, ``spectrum.metadata`` and
``dataset.peaks.metadata`` expose them.

Showing the most intense peaks
------------------------------

Use the spectrum operation when the result should remain a ``Spectrum``:

.. jupyter-input::

   top_spectrum = spectrum.sort_by_intensity()  # descending by default
   pd.DataFrame(top_spectrum.data[:10], columns=["mz", "intensity"])

Normalization is non-mutating unless ``in_place=True``:

.. jupyter-input::

   normalized = spectrum.normalize(scale=100.0)
   normalized.intensity.max()

At dataset scale, ``dataset.peaks.normalize(scale=100.0)`` normalizes every
spectrum independently. Record methods provide the same operations while
preserving record metadata.

Sorting peaks by m/z
--------------------

.. jupyter-input::

   mz_sorted = spectrum.sort_by_mz(ascending=True)
   mz_sorted.mz

``Spectrum.sort_by_mz``, ``sort_by_intensity``, and ``normalize`` default to
``in_place=False``. The corresponding ``PeakSeries`` operations can process
all visible spectra. Peak metadata is reordered with the peaks.
