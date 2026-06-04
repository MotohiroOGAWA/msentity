Loading datasets
================

This page explains how to load mass spectrometry datasets with :mod:`msentity`.

Importing msentity
------------------

.. code-block:: python

   import msentity

The most general loader is :func:`msentity.load_ms_dataset`.

.. code-block:: python

   from msentity import load_ms_dataset

Loading MSP files
-----------------

.. code-block:: python

   dataset = load_ms_dataset("example.msp")

   dataset

Example output:

.. code-block:: text

   MSDataset(n_spectra=3, n_peaks=12, columns=['Name', 'PrecursorMZ', 'IonMode', 'AdductType'])

Loading MGF files
-----------------

.. code-block:: python

   dataset = load_ms_dataset("example.mgf")

   dataset

Loading MSDS files
------------------

``.msds`` is the native dataset format used by msentity.

.. code-block:: python

   dataset = load_ms_dataset("example.msds")

   dataset

Specifying the file type manually
---------------------------------

If the file extension is not enough to determine the file type, specify
``file_type`` explicitly.

.. code-block:: python

   dataset = load_ms_dataset(
       "example.txt",
       file_type="msp",
   )

Supported file types are:

.. code-block:: text

   msp
   mgf
   msds

Generating SpecID values
------------------------

If needed, ``SpecID`` values can be generated during loading by passing
``spec_id_prefix``.

.. code-block:: python

   dataset = load_ms_dataset(
       "example.msp",
       spec_id_prefix="example",
   )

   dataset.metadata.head()