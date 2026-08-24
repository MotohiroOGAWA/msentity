<h1><img src="media/icon.png" alt="msentity icon" width="48" height="48" align="center"> msentity Spectrum Viewer for VS Code</h1>

This extension is the graphical viewer included in the
[`msentity`](https://github.com/MotohiroOGAWA/msentity) repository. It opens
`.msds`, `.msp`, and `.mgf` datasets without Gradio.

## Features

- Paged spectrum metadata table and one reusable spectrum panel
- MSP/MGF loading progress with bytes, percentage, and spectrum count
- Full-dataset export to MSDS, MSP, or MGF through the VS Code save dialog
- msentity file-type icon for `.msds`, `.msp`, and `.mgf` tabs
- one transparent PNG for the extension listing, file type, and editor tabs
- Optional VS Code floating plot window
- Peak table, m/z/intensity sorting, and peak selection
- Independent horizontal, vertical, and two-dimensional drag-to-zoom
- m/z range starting at zero and adaptive 1, 2, 2.5, 5, 10 tick spacing
- Adaptive zero-based intensity ticks
- Export preview with optional grid lines, tick numbers, and peak m/z labels
- Transparent PNG and SVG plot export (all optional elements off by default)
- VS Code light, dark, and high-contrast theme support

## Download and install the latest release

Download the
[latest `msentity-spectrum-viewer.vsix`](https://github.com/MotohiroOGAWA/msentity/releases/latest/download/msentity-spectrum-viewer.vsix).
This URL always points to the newest release, so no version needs to be
specified. Versioned files remain available from individual release pages when
a specific version is required.

Install it from a terminal:

```console
code --install-extension ./msentity-spectrum-viewer.vsix
```

Or open VS Code's Extensions view, choose **Views and More Actions (...) →
Install from VSIX...**, and select the downloaded file. Then run **Developer:
Reload Window**.

The configured Python environment must contain `msentity`:

```console
python -c "import msentity; print(msentity.__file__)"
```

Set `msentitySpectrumViewer.pythonPath` if that interpreter is not available as
`python` from VS Code.

## Build a versioned VSIX

From a clone of `msentity`:

```console
cd vscode-extension
npm ci
npm run package
```

The VSIX is written to
`dist/msentity-spectrum-viewer-<version>.vsix`; `<version>` is read from
`package.json`. The ignored `dist/` directory keeps generated packages separate
from extension source files. Test it by substituting the generated version:

```console
code --install-extension ./dist/msentity-spectrum-viewer-<version>.vsix --force
```

`npm ci` uses the committed lock file for a reproducible dependency install.
`npm run package` invokes the official VS Code extension packager. The
`.vscodeignore` file excludes development-only files from the VSIX.

For a GitHub release, run `npm run package:release`. It creates both the
versioned VSIX and `dist/msentity-spectrum-viewer.vsix` under `dist/`; upload
both assets to the release. The fixed filename is required by the recommended `latest/download`
URL, while the versioned filename supports pinned downloads.

## Usage

Open an `.msds`, `.msp`, or `.mgf` file. Click a spectrum button in the table;
the reusable **Mass Spectrum** panel opens and updates when another record is
selected. Drag horizontally to zoom m/z only, vertically to zoom intensity
only, or diagonally to zoom both axes. Choose **Export image...**, select the
elements to include, check the preview, and save the visible plot as a
transparent PNG or SVG. Tick grid lines are excluded by default.
The exported image uses the current zoom range and leaves 10% intensity
headroom so labels above maximum-intensity peaks remain readable.
When tick numbers are disabled, export compacts the axis-title spacing and
unused margins. Click empty plot space to clear peak selection; double-click
the plot to reset both zoom axes.

Opening normally detects MSDS, MSP, or MGF from the extension. To override it,
right-click a file in Explorer and choose **MS Entity: Open as MSDS**, **Open as
MSP**, or **Open as MGF**. The same commands are available in the Command
Palette. Dataset **Export...** first asks for MSDS, MSP, or MGF and then for the
save location; the chosen format takes precedence and its extension is applied
automatically.

Set `msentitySpectrumViewer.spectrumFloatingWindow` to `false` if the spectrum
should remain in an editor pane beside the dataset table.

## Settings

- `msentitySpectrumViewer.pythonPath`: Python executable used to load datasets
  (default: `python`).
- `msentitySpectrumViewer.pageSize`: spectra per metadata page (default: `20`).
- `msentitySpectrumViewer.spectrumFloatingWindow`: use a separate floating
  window for the spectrum (default: `true`).
