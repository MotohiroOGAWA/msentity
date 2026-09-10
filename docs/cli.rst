Command-line interface
======================

Installing ``msentity`` provides the ``msentity`` command. Run
``msentity --help`` or ``msentity <command> --help`` for the complete option
list.

Dataset input
-------------

``info``, ``head``, ``convert``, and ``shell`` accept MSP, MGF, MSDS, TSV, and
CSV input. The format is inferred from the extension; use ``--file-type`` for
a missing or non-standard extension. ``--spec-id-prefix`` creates sequential
``SpecID`` values when the input has no such column::

   msentity info sample.csv
   msentity head sample.msp --num-rows 10
   msentity shell sample.tsv --spec-id-prefix sample-

Convert and merge
-----------------

``convert`` currently writes the native MSDS format::

   msentity convert input.msp output.msds

``merge-dir`` combines supported files into one MSDS dataset. By default it
examines the selected directory only. Pass ``--recursive`` for all nested
directories, or a positive depth such as ``--recursive 2``. ``--pattern``
selects a glob, and ``--add-source`` adds the relative ``path`` and
``source_index`` columns::

   msentity merge-dir spectra combined.msds --recursive --add-source
   msentity merge-dir tables combined.msds --file-type csv

Dataset metadata
----------------

``meta`` reads the description, attributes, and tags from an MSDS file without
loading the complete dataset::

   msentity meta combined.msds

Similarity by key
-----------------

``similarity-by-key`` pairs unique, non-missing key values from two datasets
and writes an ``.mssim`` result. The default key on both sides is ``SpecID``::

   msentity similarity-by-key first.msds second.msds \
       --key1 SpecID --key2 SpecID --output result.mssim

Use ``--bin-width``, ``--intensity-exponent``, and ``--max-cum-peaks`` to
control calculation. Duplicate keys are rejected; unmatched keys are skipped.
See :doc:`api/similarity` for the equivalent Python API.

Interactive shell
-----------------

``msentity shell INPUT`` starts a dataset-oriented prompt. Available commands
cover information and row previews, spectrum and peak inspection, filtering,
sorting, normalization, visible columns, SpecID and PeakID assignment,
description, attributes, tags, view reset, and export. Run ``help`` inside the
shell for the command list and ``help <command>`` for usage.
