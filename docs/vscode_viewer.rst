VS Code Spectrum Viewer
=======================

``msentity-spectrum-viewer`` is the graphical viewer shipped in the
``vscode-extension`` directory of this repository. It opens MSDS, MSP, and MGF
files as a paged metadata table and displays the selected mass spectrum in one
reusable VS Code panel. It does not use or require Gradio.

Install a release VSIX
----------------------

1. Download the `latest msentity-spectrum-viewer.vsix
   <https://github.com/MotohiroOGAWA/msentity/releases/latest/download/msentity-spectrum-viewer.vsix>`_.
   This URL always points to the newest release, so no version needs to be
   specified.
2. Install the downloaded file with either method below.

From a terminal:

.. code-block:: bash

   code --install-extension ./msentity-spectrum-viewer.vsix

From VS Code, open **Extensions**, choose **Views and More Actions (...)**,
select **Install from VSIX...**, and choose the downloaded file. Run
**Developer: Reload Window** when installation finishes.

The extension starts a small Python process to read each dataset. Install
``msentity`` into that environment and verify it before opening a file:

.. code-block:: bash

   python -m pip install "msentity @ git+https://github.com/MotohiroOGAWA/msentity.git"
   python -c "import msentity; print(msentity.__file__)"

If VS Code uses a different interpreter, set
``msentitySpectrumViewer.pythonPath`` to its absolute path.

Use the viewer
--------------

Open an ``.msds``, ``.msp``, or ``.mgf`` file normally, or right-click it in
Explorer and choose **MS Entity: Open Spectrum Viewer**. Click the spectrum
button in a table row to update the spectrum panel.

Normal opening detects the input format from the filename extension. To select
it explicitly, right-click any file in Explorer and choose **MS Entity: Open as
MSDS**, **MS Entity: Open as MSP**, or **MS Entity: Open as MGF**. These commands
are also available in the Command Palette and take precedence over the filename
extension.

While an MSP or MGF file is being parsed, the editor displays a progress bar
with the percentage, processed bytes, and successfully loaded spectrum count.
The same progress display is used when **Reload** rereads the file.

Choose **Export...** to save the entire loaded dataset as MSDS, MSP, or MGF.
First select the output format explicitly, then select the save location. The
matching filename extension is applied automatically. Export always writes the
full dataset, not only the current page or filtered table rows.

.. figure:: _static/images/gui_screenshot_1.png
   :alt: VS Code custom editor showing the paged msentity dataset table
   :align: center

   Browse, filter, page through, and select spectra from the dataset table.

.. figure:: _static/images/gui_screenshot_2.png
   :alt: VS Code mass-spectrum panel with peak table and metadata
   :align: center

   Inspect the selected spectrum, peak list, and spectrum metadata together.

The plot begins at m/z 0. Drag horizontally to zoom only the m/z axis, drag
vertically to zoom only the intensity axis, or drag diagonally to zoom both.
A movement component below the drag threshold leaves that axis unchanged, so
horizontal navigation never rescales intensity. Tick spacing is recalculated
for the visible range using readable 1, 2, 2.5, 5, and 10 multiples. Click
**Reset zoom**, or double-click inside the plot, to return both axes to the
full range. Clicking empty plot space clears the selected peak. The
intensity axis always begins at zero, leaves 10% headroom above the highest
peak, and uses the same adaptive tick scheme.

Choose **Export image...** to preview the current zoomed plot before saving.
The preview can independently show or hide tick grid lines, tick numbers, and
m/z labels above each peak. All three are off by default. When tick numbers are
hidden, the axis titles move closer to their axes and the unused margins are
cropped. Choose **Save PNG** or **Save SVG** after checking the preview. Both
formats use a transparent background. PNG output is rendered at twice the SVG
dimensions; SVG output remains scalable. Export uses the currently visible m/z range,
including any active zoom.

Settings
--------

``msentitySpectrumViewer.pythonPath``
   Python executable that imports ``msentity``. Default: ``python``.

``msentitySpectrumViewer.pageSize``
   Number of spectra loaded into one metadata page. Default: 20; range: 1–500.

``msentitySpectrumViewer.spectrumFloatingWindow``
   Move the spectrum panel to a VS Code floating window. Disable it to keep the
   plot beside the table. Default: enabled.

Build the VSIX locally
----------------------

Node.js and npm are needed only to package the extension:

.. code-block:: bash

   git clone https://github.com/MotohiroOGAWA/msentity.git
   cd msentity/vscode-extension
   npm ci
   npm run package

The last command creates
``vscode-extension/dist/msentity-spectrum-viewer-<version>.vsix``, where
``<version>`` comes from ``vscode-extension/package.json``. Install and replace
an existing copy by substituting the generated version in this command:

.. code-block:: bash

   code --install-extension ./dist/msentity-spectrum-viewer-<version>.vsix --force

For a release, run ``npm run package:release`` instead. It creates both the
versioned VSIX and ``msentity-spectrum-viewer.vsix`` under ``dist/``; attach
both files to the matching GitHub release. The fixed asset name keeps the recommended
``latest/download`` URL valid while the versioned asset remains available for
pinned downloads.
