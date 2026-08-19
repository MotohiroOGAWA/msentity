SpectrumRecord
==============

.. currentmodule:: msentity

.. autoclass:: SpectrumRecord
   :members:
   :no-index:

Represents a single spectrum with associated metadata.

The :class:`~msentity.SpectrumRecord` class provides convenient access to
both spectrum-level metadata and the corresponding
:class:`~msentity.Spectrum`.

It allows direct access and modification of metadata, as well as common
operations on the spectrum such as normalization and sorting.

Metadata values use mapping syntax (``record["Name"]``). ``spectrum`` and
``peaks`` refer to the corresponding :class:`~msentity.Spectrum`, while
``n_peaks`` and ``is_integer_mz`` provide common checks.

Class
-----

Lightweight view object for accessing individual spectra.

.. autosummary::
   :toctree: generated/

   SpectrumRecord
