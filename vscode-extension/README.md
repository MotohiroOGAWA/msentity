# msentity Spectrum Viewer for VS Code

This extension is the graphical viewer included in the
[`msentity`](https://github.com/MotohiroOGAWA/msentity) repository. It opens
`.msds`, `.msp`, and `.mgf` datasets without Gradio.

## Features

- Paged spectrum metadata table and one reusable spectrum panel
- Optional VS Code floating plot window
- Peak table, m/z/intensity sorting, and peak selection
- Horizontal drag-to-zoom and Reset zoom
- m/z range starting at zero and adaptive 1, 2, 2.5, 5, 10 tick spacing
- Adaptive zero-based intensity ticks
- Transparent PNG and SVG plot export
- VS Code light, dark, and high-contrast theme support

## Download and install 0.1.1

Download `msentity-spectrum-viewer-0.1.1.vsix` from the assets of the
[`msentity-v0.1.1` release](https://github.com/MotohiroOGAWA/msentity/releases/tag/msentity-v0.1.1).

Install it from a terminal:

```console
code --install-extension ./msentity-spectrum-viewer-0.1.1.vsix
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

## Build `msentity-spectrum-viewer-0.1.1.vsix`

From a clone of `msentity`:

```console
cd vscode-extension
npm ci
npm run package
```

The VSIX is written to the current directory. Test it with:

```console
code --install-extension ./msentity-spectrum-viewer-0.1.1.vsix --force
```

`npm ci` uses the committed lock file for a reproducible dependency install.
`npm run package` invokes the official VS Code extension packager. The
`.vscodeignore` file excludes development-only files from the VSIX.

## Usage

Open an `.msds`, `.msp`, or `.mgf` file. Click a spectrum button in the table;
the reusable **Mass Spectrum** panel opens and updates when another record is
selected. Drag horizontally to zoom. Use **Save PNG** or **Save SVG** to export
the visible plot with a transparent background.

Set `msentitySpectrumViewer.spectrumFloatingWindow` to `false` if the spectrum
should remain in an editor pane beside the dataset table.

## Settings

- `msentitySpectrumViewer.pythonPath`: Python executable used to load datasets
  (default: `python`).
- `msentitySpectrumViewer.pageSize`: spectra per metadata page (default: `20`).
- `msentitySpectrumViewer.spectrumFloatingWindow`: use a separate floating
  window for the spectrum (default: `true`).
