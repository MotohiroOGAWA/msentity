Getting Started
===============

``msentity`` is a lightweight Python toolkit for handling mass spectrometry data
in a consistent and programmatic way.

It is designed for mass spectrometry workflows, providing:

- Spectrum-level metadata handling with pandas DataFrame
- Peak-level storage and access with ``PeakSeries`` and ``Spectrum``
- Dataset-level operations such as slicing, sorting, merging, and HDF5 I/O

Requirements
------------

- Python 3.10 or later
- NumPy
- pandas
- h5py
- pyarrow

Installation
------------

Install directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/MotohiroOGAWA/msentity.git

For development, you can clone the repository and install in editable mode:

.. code-block:: bash

   pip install git+https://github.com/MotohiroOGAWA/msentity.git
   cd msentity
   pip install -e .[docs]

Testing
-------

Run tests to verify the installation:

.. code-block:: bash

   python -m unittest discover -s msentity/tests -p "Test*.py" -v

Basic Usage
-----------

.. code-block:: python

   from msentity import MSDataset

   dataset = MSDataset.from_hdf5("example.h5")

   # Access a single spectrum record
   record = dataset[0]

   # Spectrum-level metadata
   print(record["Name"])
   print(record["PrecursorMZ"])

   # Peak-level data
   print(record.spectrum.mz)
   print(record.spectrum.intensity)

   # Iterate over the dataset
   for record in dataset:
       print(record["Name"], record.n_peaks)