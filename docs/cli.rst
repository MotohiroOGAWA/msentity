Command-line interface
======================

Installing ``msentity`` provides the ``msentity`` command. The examples below
use :download:`example.msp <examples/example.msp>`, whose complete contents
are shown in :doc:`getting_started`. Run commands from ``docs/examples`` or
replace the path with your own file.

Command overview
----------------

.. code-block:: console

   $ msentity --help
   usage: msentity [-h]
                   {convert,head,info,library-search,merge-dir,meta,similarity-by-key,shell}
                   ...

Use ``msentity COMMAND --help`` for every option. ``info``, ``head``,
``convert``, and ``shell`` accept MSP, MGF, MSDS, TSV, and CSV. The extension
selects the reader; ``--file-type`` overrides it. ``--spec-id-prefix sample-``
adds ``sample-1``, ``sample-2``, ... only if ``SpecID`` is absent.

Inspect a dataset
-----------------

.. code-block:: console

   $ msentity info example.msp
   {
     "input_file": "example.msp",
     "n_spectra": 3,
     "n_columns": 5,
     "n_peaks_total": 12,
     "columns": [
       "Name",
       "PrecursorMZ",
       "AdductType",
       "CollisionEnergy",
       "NumPeaks"
     ],
     "description": "",
     "attributes": {},
     "tags": []
   }

``head`` prints spectrum metadata, not peak lists:

.. code-block:: console

   $ msentity head example.msp --num-rows 3
            Name PrecursorMZ AdductType CollisionEnergy NumPeaks
   0  Compound_A    301.2162     [M+H]+              20        4
   1  Compound_B    255.1234     [M+H]+              30        3
   2  Compound_C    412.2871    [M+Na]+              25        5

Convert and inspect MSDS metadata
---------------------------------

.. code-block:: console

   $ msentity convert example.msp example.msds
   Saved: example.msds
   $ msentity meta example.msds
   {
     "description": "",
     "attributes": {},
     "tags": []
   }

``meta`` reads only dataset-level metadata. Use ``info`` to see spectrum and
peak counts.

Merge a directory
-----------------

Suppose ``spectra`` contains ``a.msp`` and ``nested/b.mgf``:

.. code-block:: console

   $ msentity merge-dir spectra combined.msds --recursive --add-source
   Merged 2 files into: combined.msds

Without ``--recursive`` only the selected directory is searched. Use
``--recursive 2`` for that directory plus its direct children. ``--pattern``
accepts a glob. ``--add-source`` creates ``path`` and ``source_index`` columns;
verify them with ``msentity head combined.msds``.

Similarity commands
-------------------

The default matching key is ``SpecID`` on both sides. Keys must be unique and
nonempty; unmatched keys are skipped.

.. code-block:: console

   $ msentity similarity-by-key first.msds second.msds \
       --key1 SpecID --key2 SpecID --output result.mssim
   Saved 3 similarities: result.mssim
   $ msentity library-search query.msds reference.msp \
       --method cosine --threshold 0.8 --output matches.mssim
   Saved 17 library matches: matches.mssim

The printed counts depend on the inputs. Key matching is lightweight by
default; add ``--include-matched-data`` to embed spectra. Library search embeds
matches by default; add ``--lightweight`` to omit them. Both support cosine or
reverse cosine, bin width, intensity exponent, and chunk controls. See
:doc:`api/similarity` for the Python API.

Interactive shell: input and output
-----------------------------------

The following is a shortened real session. Commands follow ``msentity>`` and
the remaining lines are their output.

.. code-block:: console

   $ msentity shell example.msp
   msentity shell
   Type 'help' to show commands. Type 'exit' to quit.

   msentity> info
   input_file: example.msp
   n_spectra: 3
   n_columns: 5
   n_peaks_total: 12
   description:
   attributes: {}
   tags: []
   msentity> show 0 --top 2 --sort intensity
   Metadata
   --------
   Name               Compound_A
   PrecursorMZ          301.2162
   AdductType             [M+H]+
   CollisionEnergy            20
   NumPeaks                    4

   Peaks
   -----
      mz  intensity
   301.2      1.000
   145.1      0.553
   msentity> filter PrecursorMZ > 300
   Filtered dataset: 2 spectra
   msentity> sort PrecursorMZ desc
   Sorted by PrecursorMZ (desc)
   msentity> head 5
            Name PrecursorMZ AdductType CollisionEnergy NumPeaks
   0  Compound_C    412.2871    [M+Na]+              25        5
   1  Compound_A    301.2162     [M+H]+              20        4
   msentity> specid --prefix SP
   Assigned spectrum IDs: column='SpecID', prefix='SP', start=1
   msentity> export selected.msds
   Saved: selected.msds
   msentity> exit

Shell commands also manage visible columns, normalization, peak IDs,
description, attributes, tags, and view reset. Run ``help`` for the current
list and ``help COMMAND`` for syntax and examples. Changes live in memory until
``export`` is run; ``reset`` resets the view but does not undo edits.
