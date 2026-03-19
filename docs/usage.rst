Usage
=====

This page shows common operations for working with :class:`msentity.MSDataset`.

In practice, datasets are typically loaded from files such as MSP or HDF5.

Basic iteration
---------------

You can iterate over spectra using a simple loop.

.. code-block:: python

   for record in dataset:
       print(record["name"], record.n_peaks)

Each element is a :class:`SpectrumRecord`.

Accessing a single spectrum
---------------------------

You can access a spectrum by index.

.. code-block:: python

   record = dataset[0]

   record["name"]
   record["precursor_mz"]
   record.n_peaks

Accessing peak data
-------------------

Each record provides access to its spectrum.

.. code-block:: python

   spectrum = record.spectrum

   spectrum.mz
   spectrum.intensity

Iterating over peaks:

.. code-block:: python

   for peak in spectrum:
       print(peak.mz, peak.intensity)

Accessing metadata
------------------

Spectrum-level metadata:

.. code-block:: python

   dataset["name"]
   dataset["precursor_mz"]

Peak-level metadata (via spectrum):

.. code-block:: python

   spectrum["annotation"]

Adding metadata
---------------

You can add new metadata columns.

.. code-block:: python

   dataset["score"] = [0.91, 0.83, 0.75]

Or assign a constant value:

.. code-block:: python

   dataset["split"] = "train"

Slicing and filtering
---------------------

You can create subsets using slicing.

.. code-block:: python

   subset = dataset[0:10]

Using index lists:

.. code-block:: python

   subset = dataset[[0, 2, 5]]

Using boolean masks:

.. code-block:: python

   subset = dataset[dataset["precursor_mz"] > 200]

Sorting
-------

Sort by a metadata column.

.. code-block:: python

   dataset = dataset.sort_by("precursor_mz", ascending=False)

Spectrum operations
-------------------

Operations can be applied at the spectrum level.

.. code-block:: python

   record = dataset[0]

   record = record.normalize(scale=1.0)
   record = record.sort_by_mz()
   record = record.sort_by_intensity()

Dataset copy
------------

Create an independent copy of the dataset.

.. code-block:: python

   new_dataset = dataset.copy()

HDF5 I/O
--------

Save and load datasets.

.. code-block:: python

   dataset.to_hdf5("data.h5")

   dataset = MSDataset.from_hdf5("data.h5")