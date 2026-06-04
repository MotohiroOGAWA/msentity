Usage
=====

This section introduces common workflows with :mod:`msentity` from Python.

The examples are written as small, reusable Python snippets. They can be run in
scripts, interactive Python sessions, Jupyter Notebook, or JupyterLab.

Most operations shown here correspond to operations available from the
``msentity`` command line interface.

:class:`msentity.MSDataset` is the central object in msentity. It represents a
collection of mass spectra together with spectrum-level metadata and peak
arrays.

Typical workflow
----------------

A common workflow is:

1. Load an MSP, MGF, or MSDS file.
2. Inspect dataset-level information.
3. Preview spectrum metadata.
4. Inspect individual spectra and peaks.
5. Filter spectra by metadata.
6. Save the result as an ``.msds`` file.

Pages
-----

.. toctree::
   :maxdepth: 1

   loading
   inspecting
   records_and_peaks
   filtering
   saving
   metadata

Relationship to the CLI
-----------------------

The Python API can reproduce the main CLI operations.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - CLI command
     - Python equivalent
   * - ``msentity info input.msp``
     - ``load_ms_dataset("input.msp")`` and inspect ``len(dataset)``, ``dataset.n_columns``, and ``dataset.n_peaks_total``
   * - ``msentity head input.msp -n 10``
     - ``dataset.metadata.head(10)``
   * - ``msentity convert input.msp output.msds``
     - ``dataset = load_ms_dataset("input.msp")`` followed by ``dataset.save("output.msds")``
   * - ``msentity meta input.msds``
     - ``MSDataset.read_dataset_meta("input.msds")``
   * - ``show 0`` in ``msentity shell``
     - ``dataset.metadata.iloc[0]`` or ``dataset[0]``
   * - ``peaks 0 --top 10 --sort intensity``
     - Convert ``dataset.peaks[0]`` to a pandas DataFrame and sort by ``intensity``
   * - ``filter PrecursorMZ > 300``
     - ``dataset[dataset["PrecursorMZ"] > 300]``

Recommended reading order
-------------------------

If you are new to msentity, start with :doc:`loading`, then continue with
:doc:`inspecting` and :doc:`records_and_peaks`.

For preprocessing workflows, read :doc:`filtering` and :doc:`saving`.

For dataset-level information such as descriptions, attributes, and tags, see
:doc:`metadata`.