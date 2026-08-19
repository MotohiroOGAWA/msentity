Spectrum
========

.. currentmodule:: msentity

.. autoclass:: Spectrum
   :members:
   :no-index:

Represents a single mass spectrum.

The :class:`~msentity.Spectrum` class provides access to m/z and intensity
arrays and supports common operations such as normalization, sorting,
and peak inspection.

``data`` is a two-column NumPy array (m/z, intensity). ``metadata`` contains
optional peak annotations. ``normalize``, ``sort_by_mz``, and
``sort_by_intensity`` return an independent spectrum by default and mutate only
when ``in_place=True``.

Class
-----

Lightweight wrapper for individual spectrum operations.

.. autosummary::
   :toctree: generated/

   Spectrum
