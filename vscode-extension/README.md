<h1><img src="media/icon.png" alt="msentity icon" width="48" height="48" align="center"> msentity Spectrum Viewer for VS Code</h1>

This extension is the graphical viewer included in the
[`msentity`](https://github.com/MotohiroOGAWA/msentity) repository. It opens
`.msds`, `.msp`, and `.mgf` datasets without Gradio.

## Features

- Paged spectrum metadata table with reorderable columns and one reusable spectrum panel
- Full-dataset, multi-condition filtering (separate text/number `=`, text `!=`, `contains`, `>`, `>=`, `<`, `<=`) and prioritized multi-column row sorting
- MSP/MGF loading progress with bytes, percentage, and spectrum count
- Export to MSDS, MSP, or MGF using the visible column order and current row filters/sort order
- msentity file-type icon for `.msds`, `.msp`, and `.mgf` tabs
- one transparent PNG for the extension listing, file type, and editor tabs
- Optional VS Code floating plot window
- Peak table, m/z/intensity sorting, and peak selection
- Copy or save the current upper spectrum's complete peak list as TSV
- Independent horizontal, vertical, and two-dimensional drag-to-zoom
- m/z range starting at zero and adaptive 1, 2, 2.5, 5, 10 tick spacing
- Adaptive zero-based intensity ticks
- Multiple datasets in one viewer with a dataset dropdown and per-dataset page memory
- Mirrored two-spectrum comparison with independently pinnable upper/lower slots
- Dot product, reverse dot product, modified dot product, and BONANZA similarity
- Export preview with optional grid lines, separate m/z/intensity tick numbers,
  and peak m/z labels
- Custom-sized transparent PNG/SVG export and clipboard copy
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
selected. Use **Add dataset…** to load more files into the same dataset viewer,
then switch between them with the dataset dropdown. Spectra opened from those
datasets share the same Mass Spectrum panel, while datasets opened in separate
VS Code tabs continue to use separate spectrum panels. Each dataset remembers
its current page when you switch away and return.

Pin the upper spectrum and select another spectrum to compare them on a shared
m/z axis. The second spectrum is drawn downward in red. Each side can be pinned
independently: new selections replace the unpinned side, and neither side
changes while both are pinned. The lower spectrum can also be removed. While
comparing, similarity is calculated automatically with a selectable dot
product, reverse dot product, modified dot product, or BONANZA score. Fragment
matching tolerance is editable in Da and defaults to ±0.05 Da. Modified dot
product and BONANZA also consider precursor-mass-shifted neutral-loss matches.
The score and matched-peak count update when the method or tolerance changes.
Drag horizontally to zoom m/z only, vertically to zoom intensity
only, or diagonally to zoom both axes. Choose **Export image...**, select the
elements to include, set any output width and height in pixels, and save the
visible plot as a transparent PNG or SVG. The requested dimensions are filled
even when their aspect ratio differs from the plot. **Copy PNG** and **Copy
SVG** copy the current preview to the clipboard. Tick grid lines are excluded
by default. Plot geometry follows the requested aspect ratio, while axis,
tick, and peak-label text keeps its original size and character proportions.
The exported image uses the current zoom range and leaves 10% intensity
headroom so labels above maximum-intensity peaks remain readable.
When both kinds of tick numbers are disabled, export compacts the axis-title
spacing and unused margins. Click empty plot space to clear peak selection; double-click
the plot to reset both zoom axes.

Use **Copy TSV** above the peak table to copy all peaks from the current upper
spectrum with `m/z` and `Intensity` columns. **Save TSV** writes the same
unrounded values to a tab-separated file. These actions include the complete
spectrum, independent of the current plot zoom, in the peak table's current
m/z or Intensity sort order.

Opening normally detects MSDS, MSP, or MGF from the extension. To override it,
right-click a file in Explorer and choose **MS Entity: Open as MSDS**, **Open as
MSP**, or **Open as MGF**. The same commands are available in the Command
Palette. Dataset **Export...** first asks for MSDS, MSP, or MGF and then for the
save location; the chosen format takes precedence and its extension is applied
automatically. The exported dataset contains the currently selected columns in
their displayed order and the rows in their current filtered and sorted order.

Set `msentitySpectrumViewer.spectrumFloatingWindow` to `false` if the spectrum
should remain in an editor pane beside the dataset table.

## Settings

- `msentitySpectrumViewer.pythonPath`: Python executable used to load datasets
  (default: `python`).
- `msentitySpectrumViewer.pageSize`: spectra per metadata page (default: `20`).
- `msentitySpectrumViewer.spectrumFloatingWindow`: use a separate floating
  window for the spectrum (default: `true`).
