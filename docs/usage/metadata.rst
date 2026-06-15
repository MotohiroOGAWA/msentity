Dataset metadata
================

This page explains dataset-level metadata such as description, attributes, and
tags.

Loading a dataset
-----------------

.. code-block:: python

   from msentity import load_ms_dataset

   dataset = load_ms_dataset("example.msp")

Description
-----------

.. code-block:: python

   dataset.description = "Example dataset for msentity usage"

   dataset.description

Attributes
----------

Attributes are key-value pairs stored at the dataset level.

.. code-block:: python

   dataset.set_attribute("source", "example")
   dataset.set_attribute("instrument", "LC-MS/MS")

   dataset.attributes

Removing an attribute:

.. code-block:: python

   dataset.remove_attribute("source")

   dataset.attributes

Tags
----

Tags are simple labels attached to the dataset.

.. code-block:: python

   dataset.add_tag("demo")
   dataset.add_tag("msp")

   dataset.tags

Removing a tag:

.. code-block:: python

   dataset.remove_tag("demo")

   dataset.tags

Reading metadata from MSDS
--------------------------

The CLI command:

.. code-block:: bash

   msentity meta example.msds

corresponds to:

.. code-block:: python

   from msentity import MSDataset

   meta = MSDataset.read_dataset_meta("example.msds")

   meta

You can access each field.

.. code-block:: python

   meta.description
   meta.attributes
   meta.tags