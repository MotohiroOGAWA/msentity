Usage
=====

This page introduces common workflows with :class:`msentity.MSDataset`.

An :class:`msentity.MSDataset` represents a collection of spectra together with
their associated metadata. It can be used as:

- a dataset container for many spectra
- an iterator over spectrum records
- a table-like object for metadata access and filtering
- a convenient interface for spectral peak arrays such as m/z and intensity

In a typical workflow, a dataset is loaded from an MSP or MGF file, inspected,
filtered, processed, and then written back to a file.

Loading datasets
----------------

Datasets can be loaded from MSP or MGF files.

.. code-block:: python

   from msentity import read_msp, read_mgf

   dataset = read_msp("example.msp")
   dataset2 = read_mgf("example.mgf")

Printing the dataset usually shows a short summary.

.. code-block:: python

   print(dataset)

Example output:

.. code-block:: text

   MSDataset(n_spectra=3, n_columns=4, n_peaks_total=12)

This tells you:

- how many spectra are stored
- how many metadata columns are available
- how many peaks are stored in total

Inspecting the dataset
----------------------

You can inspect basic dataset-level properties.

.. code-block:: python

   print(dataset.n_columns)
   print(dataset.n_peaks_total)
   print(dataset.columns)
   print(dataset.description)

Example output:

.. code-block:: text

   4
   12
   ['name', 'precursor_mz', 'adduct', 'collision_energy']
   Example MSP dataset

If the dataset has metadata, it can be accessed in a table-like way.

.. code-block:: python

   print(dataset["name"])

Example output:

.. code-block:: text

   ['Compound_A', 'Compound_B', 'Compound_C']

This is useful when you want to inspect one column across all spectra, similar
to selecting a column in pandas.

Iterating over spectra
----------------------

An :class:`MSDataset` can be used as an iterator over spectrum records.

.. code-block:: python

   for record in dataset:
       print(record["name"], record.n_peaks)

Example output:

.. code-block:: text

   Compound_A 4
   Compound_B 3
   Compound_C 5

Each element is a :class:`msentity.SpectrumRecord`, which represents one
spectrum together with its metadata.

Accessing a single spectrum record
----------------------------------

You can access a single record by index.

.. code-block:: python

   record = dataset[0]

   print(record)
   print(record["name"])
   print(record["precursor_mz"])
   print(record.n_peaks)

Example output:

.. code-block:: text

   SpectrumRecord(index=0, n_peaks=4)
   Compound_A
   301.2162
   4

A record behaves like one row of the dataset, while still giving direct access
to spectral data.

Accessing spectral peaks
------------------------

Each record provides access to its spectrum.

.. code-block:: python

   spectrum = record.spectrum

   print(spectrum.mz)
   print(spectrum.intensity)

Example output:

.. code-block:: text

   [100.0, 145.1, 183.2, 301.2]
   [12.0, 55.3, 21.7, 100.0]

This is the most common way to access peak arrays for plotting, matching, or
feature extraction.

You can also iterate over peaks.

.. code-block:: python

   for peak in spectrum:
       print(peak.mz, peak.intensity)

Example output:

.. code-block:: text

   100.0 12.0
   145.1 55.3
   183.2 21.7
   301.2 100.0

This is helpful when writing custom scoring or filtering logic peak by peak.

Working with metadata
---------------------

Dataset metadata can be accessed in a pandas-like style.

.. code-block:: python

   names = dataset["name"]
   precursor_mz = dataset["precursor_mz"]

   print(names)
   print(precursor_mz)

Example output:

.. code-block:: text

   ['Compound_A', 'Compound_B', 'Compound_C']
   [301.2162, 255.1234, 412.2871]

This makes it easy to inspect or process a column for all spectra at once.

Record-level metadata can also be accessed.

.. code-block:: python

   record = dataset[0]

   print(record["name"])
   print(record["precursor_mz"])

Example output:

.. code-block:: text

   Compound_A
   301.2162

Peak-level metadata can be accessed from the spectrum if available.

.. code-block:: python

   print(spectrum["annotation"])

Example output:

.. code-block:: text

   ['frag_a', 'frag_b', 'frag_c', 'precursor']

Adding or modifying metadata
----------------------------

New metadata columns can be added to the dataset.

.. code-block:: python

   dataset["score"] = [0.91, 0.83, 0.75]
   print(dataset["score"])

Example output:

.. code-block:: text

   [0.91, 0.83, 0.75]

You can also assign a constant value to all records.

.. code-block:: python

   dataset["split"] = "train"
   print(dataset["split"])

Example output:

.. code-block:: text

   ['train', 'train', 'train']

This is useful when annotating datasets for machine learning, validation splits,
or quality flags.

Selecting subsets
-----------------

You can create subsets in several ways.

Slicing:

.. code-block:: python

   subset = dataset[0:2]
   print(subset)

Example output:

.. code-block:: text

   MSDataset(n_spectra=2, n_columns=4, n_peaks_total=7)

Selecting by a list of indices:

.. code-block:: python

   subset = dataset[[0, 2]]
   print(subset["name"])

Example output:

.. code-block:: text

   ['Compound_A', 'Compound_C']

Selecting by a boolean mask:

.. code-block:: python

   subset = dataset[dataset["precursor_mz"] > 300]
   print(subset["name"])

Example output:

.. code-block:: text

   ['Compound_A', 'Compound_C']

These operations are especially useful in data analysis workflows where spectra
must be filtered by precursor m/z, adduct type, annotation status, or score.

Sorting
-------

Datasets can be sorted by a metadata column.

.. code-block:: python

   sorted_dataset = dataset.sort_by("precursor_mz", ascending=False)
   print(sorted_dataset["precursor_mz"])

Example output:

.. code-block:: text

   [412.2871, 301.2162, 255.1234]

Sorting is useful before visualization, exporting ranked candidates, or checking
data consistency.

Common spectral operations
--------------------------

Operations can also be applied to individual records or spectra.

.. code-block:: python

   record = dataset[0]

   normalized = record.normalize(scale=1.0)
   by_mz = record.sort_by_mz()
   by_intensity = record.sort_by_intensity()

   print(normalized.spectrum.intensity)
   print(by_mz.spectrum.mz)
   print(by_intensity.spectrum.intensity)

Example output:

.. code-block:: text

   [0.12, 0.553, 0.217, 1.0]
   [100.0, 145.1, 183.2, 301.2]
   [100.0, 55.3, 21.7, 12.0]

Such operations are common before spectrum comparison, similarity scoring, or
visualization.

Copying datasets
----------------

You can create an independent copy of a dataset.

.. code-block:: python

   new_dataset = dataset.copy()

   print(new_dataset is dataset)
   print(new_dataset["name"])

Example output:

.. code-block:: text

   False
   ['Compound_A', 'Compound_B', 'Compound_C']

This is useful when you want to modify a dataset without changing the original
object.

Writing datasets
----------------

Datasets can be written back to MSP or MGF files.

.. code-block:: python

   from msentity import write_msp, write_mgf

   write_msp(dataset, "output.msp")
   write_mgf(dataset, "output.mgf")

A typical workflow is:

.. code-block:: python

   from msentity import read_msp, write_msp

   dataset = read_msp("input.msp")
   filtered = dataset[dataset["precursor_mz"] > 300]
   write_msp(filtered, "filtered.msp")

This makes it easy to perform file-based preprocessing pipelines.

Typical analysis workflow
-------------------------

The following example shows a common data analysis pattern.

.. code-block:: python

   from msentity import read_msp

   dataset = read_msp("example.msp")

   print(dataset)
   print(dataset["name"])

   filtered = dataset[dataset["precursor_mz"] > 300]
   filtered = filtered.sort_by("precursor_mz", ascending=False)

   for record in filtered:
       print(record["name"], record["precursor_mz"], record.n_peaks)

Example output:

.. code-block:: text

   MSDataset(n_spectra=3, n_columns=4, n_peaks_total=12)
   ['Compound_A', 'Compound_B', 'Compound_C']
   Compound_C 412.2871 5
   Compound_A 301.2162 4

This workflow combines:

- file loading
- metadata inspection
- filtering
- sorting
- iteration over selected spectra

These are the most common operations in exploratory spectral data analysis.

See also
--------

- :func:`msentity.read_msp`
- :func:`msentity.read_mgf`
- :func:`msentity.write_msp`
- :func:`msentity.write_mgf`
- :class:`msentity.MSDataset`
- :class:`msentity.SpectrumRecord`
- :class:`msentity.Spectrum`