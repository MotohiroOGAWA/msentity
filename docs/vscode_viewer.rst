VS Code Spectrum Viewer
=======================

``msentity-spectrum-viewer`` is the graphical viewer shipped in the
``vscode-extension`` directory of this repository. It opens MSDS, MSP, and MGF
files as a paged metadata table and displays the selected mass spectrum in one
reusable VS Code panel. It does not use or require Gradio.

Install a release VSIX
----------------------

1. Open the `msentity releases page
   <https://github.com/MotohiroOGAWA/msentity/releases>`_.
2. Open release ``msentity-v0.1.1`` and download
   ``msentity-spectrum-viewer-0.1.1.vsix`` from **Assets**.
3. Install the downloaded file with either method below.

From a terminal:

.. code-block:: bash

   code --install-extension ./msentity-spectrum-viewer-0.1.1.vsix

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

The plot begins at m/z 0. Drag horizontally across the plot to zoom; tick
spacing is recalculated for the visible range using readable 1, 2, 2.5, 5,
and 10 multiples. Click **Reset zoom** to return to the full range. The
intensity axis always begins at zero and uses the same adaptive tick scheme.

Choose **Export image...** to preview the current zoomed plot before saving.
The preview can independently show or hide tick grid lines, tick numbers, and
m/z labels above each peak. Grid lines and peak labels are off by default;
tick numbers are on. Choose **Save PNG** or **Save SVG** after checking the
preview. Both formats use a transparent background. PNG output is rendered at
1800 × 1080; SVG output remains scalable.

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
``vscode-extension/msentity-spectrum-viewer-0.1.1.vsix``. Install and replace
an existing copy with:

.. code-block:: bash

   code --install-extension ./msentity-spectrum-viewer-0.1.1.vsix --force

For a release, attach that exact VSIX file to the matching GitHub release so
users do not need Node.js or the source checkout.
