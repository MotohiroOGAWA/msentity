Loading datasets
================

This page explains file and in-memory loading with :mod:`msentity`.  MSP and
MGF are text exchange formats; MSDS is the native HDF5-backed format that also
preserves dataset-level and peak-level metadata.

Importing msentity
------------------

.. jupyter-input::

   import msentity
   from msentity import load_ms_dataset

``load_ms_dataset`` dispatches by filename extension and returns an
:class:`msentity.MSDataset`.

Loading MSP files
-----------------

.. jupyter-input::

   dataset = load_ms_dataset("example.msp")
   dataset

Typical MSP fields are canonicalized (for example, ``PRECURSORMZ`` becomes
``PrecursorMZ``). Unknown fields are retained rather than discarded. Peak
lines may also contain annotations, which become peak-level metadata.

Loading MGF files
-----------------

.. jupyter-input::

   dataset = load_ms_dataset("example.mgf")
   dataset.metadata.head()

Each ``BEGIN IONS`` block becomes one spectrum. Header keys are canonicalized,
and the peak list is stored in ``dataset.peaks``.

Loading MSDS files
------------------

``.msds`` is the native format. ``.h5`` and ``.hdf5`` extensions are accepted
by the general loader as aliases.

.. jupyter-input::

   dataset = load_ms_dataset("example.msds")

   # Equivalent, with control over peak metadata loading:
   dataset = msentity.MSDataset.load(
       "example.msds",
       load_peak_metadata=True,
   )

Specifying the file type manually
---------------------------------

Pass ``file_type`` when the extension is missing or non-standard:

.. jupyter-input::

   dataset = load_ms_dataset("example.txt", file_type="msp")

Valid values are ``"msp"``, ``"mgf"``, and ``"msds"``. An unknown extension
without ``file_type`` raises :class:`ValueError`.

Generating SpecID values
------------------------

``spec_id_prefix`` creates ``SpecID`` values only when that column is absent:

.. jupyter-input::

   dataset = load_ms_dataset(
       "example.msp",
       spec_id_prefix="sample-",
   )
   dataset["SpecID"].head()

For an already loaded dataset, use
:func:`msentity.processing.id.set_spec_id`. MSP and MGF content can also be
parsed directly from strings with :func:`msentity.read_msp_text` and
:func:`msentity.read_mgf_text`, which is useful for uploads and web APIs.
