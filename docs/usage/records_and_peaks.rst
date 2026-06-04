Records and peaks
=================

This page explains how to inspect individual spectra and peak arrays.

Loading a dataset
-----------------

.. code-block:: python

   import pandas as pd
   from msentity import load_ms_dataset

   dataset = load_ms_dataset("example.msp")

Accessing one spectrum record
-----------------------------

The interactive shell command:

.. code-block:: text

   show 0

corresponds to:

.. code-block:: python

   dataset.metadata.iloc[0]

You can also access a :class:`msentity.SpectrumRecord`.

.. code-block:: python

   record = dataset[0]

   record

Example output:

.. code-block:: text

   SpectrumRecord(index=0, n_peaks=4)

Record-level metadata
---------------------

.. code-block:: python

   record["Name"]

.. code-block:: python

   record["PrecursorMZ"]

Accessing peaks
---------------

The interactive shell command:

.. code-block:: text

   peaks 0

corresponds to:

.. code-block:: python

   spectrum = dataset.peaks[0]

   peaks = pd.DataFrame(
       spectrum.data,
       columns=["mz", "intensity"],
   )

   peaks

Example output:

.. code-block:: text

          mz  intensity
   0  100.0       12.0
   1  145.1       55.3
   2  183.2       21.7
   3  301.2      100.0

Showing the most intense peaks
------------------------------

The shell command:

.. code-block:: text

   peaks 0 --top 10 --sort intensity

corresponds to:

.. code-block:: python

   peaks.sort_values(
       "intensity",
       ascending=False,
   ).head(10)

Sorting peaks by m/z
--------------------

.. code-block:: python

   peaks.sort_values(
       "mz",
       ascending=True,
   )