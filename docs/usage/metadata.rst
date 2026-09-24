Dataset metadata
================

Dataset-level metadata describes the collection as a whole and is distinct
from spectrum metadata (DataFrame columns) and peak metadata.

Loading a dataset
-----------------

.. jupyter-input::

   from msentity import load_ms_dataset
   dataset = load_ms_dataset("example.msp")

Description
-----------

.. jupyter-input::

   dataset.description = "Curated positive-mode reference spectra"
   dataset.description

.. code-block:: text

   'Curated positive-mode reference spectra'

The value must be a string and is preserved by MSDS save/load.

Attributes
----------

Attributes are string-to-string key/value pairs:

.. jupyter-input::

   dataset.set_attribute("source", "MassBank")
   dataset.set_attribute("instrument", "Orbitrap")
   dataset.has_attribute("source"), dataset.attributes

.. code-block:: text

   (True, {'source': 'MassBank', 'instrument': 'Orbitrap'})

``set_attribute`` updates an existing key. ``remove_attribute`` returns
whether a key existed, and ``clear_attributes`` removes every attribute.
The ``attributes`` getter returns a copy, so mutate through these methods or
assign a complete dictionary.

Removing an attribute:

.. jupyter-input::

   removed = dataset.remove_attribute("source")
   removed, dataset.attributes

.. code-block:: text

   (True, {'instrument': 'Orbitrap'})

Tags
----

Tags are unique string labels. ``add_tag`` and ``remove_tag`` report whether
they changed the dataset:

.. jupyter-input::

   dataset.add_tag("reference")
   dataset.add_tag("positive-mode")
   dataset.has_tag("reference"), dataset.tags

.. code-block:: text

   (True, ['reference', 'positive-mode'])

Removing a tag:

.. jupyter-input::

   dataset.remove_tag("reference")
   dataset.tags

.. code-block:: text

   ['positive-mode']

Use ``clear_tags`` to remove all tags. Assigning ``dataset.tags`` replaces the
complete ordered list.

Reading metadata from MSDS
--------------------------

Read only the small dataset metadata group without loading spectra or peaks:

.. code-block:: bash

   msentity meta example.msds

.. jupyter-input::

   from msentity import MSDataset
   meta = MSDataset.read_dataset_meta("example.msds")
   meta

The result is an immutable ``MSDatasetMeta`` dataclass:

.. jupyter-input::

   meta.description
   meta.attributes
   meta.tags

For spectrum-level enrichment, ``dataset.merge_metadata(table, on="SMILES")``
left-joins a DataFrame into the current view. ``add_columns``, ``right_prefix``,
``overwrite``, and duplicate-handling options make the join explicit.
