<h1><img src="media/icon.png" alt="msentity icon" width="48" height="48" align="center"> msentity Spectrum Viewer for VS Code</h1>

This extension is the graphical viewer included in the
[`msentity`](https://github.com/MotohiroOGAWA/msentity) repository. It opens
MSDS, MSP, MGF, TSV, and CSV datasets without Gradio.

## Features

- Paged spectrum metadata table with reorderable columns and one reusable spectrum panel
- Full-dataset, multi-condition filtering (separate text/number `=`, text `!=`, `contains`, `>`, `>=`, `<`, `<=`) and prioritized multi-column row sorting
- Assign sequential SpecID values from the toolbar with an optional prefix
- MSP/MGF loading progress with bytes, percentage, and spectrum count
- Export to MSDS, MSP, MGF, TSV, or CSV using the visible column order and current row filters/sort order
- msentity file-type icon for `.msds`, `.msp`, and `.mgf` tabs
- one transparent PNG for the extension listing, file type, and editor tabs
- Optional VS Code floating plot window
- Peak table, m/z/intensity sorting, and peak selection
- Copy or save the complete peak table as TSV or CSV
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

## Calculate similarity and run a library search

Click **Calculate similarity…** next to **Reload**, then choose **Library
search** or **Match by metadata key**. A library search compares every spectrum
in the query dataset with every spectrum in a reference library. The reference
can be an entry already loaded with **Add dataset…**, or an MSDS, MSP, MGF, TSV,
or CSV file selected for that calculation. Choose cosine or reverse cosine,
set the score threshold (default `0.8`), and configure the NumPy binning and
chunk parameters before saving the result as **`.mssim`**.

Metadata-key matching preserves the previous workflow. Select two loaded
datasets and a key for each side. Non-missing keys must be unique within each
dataset. Duplicate values produce an error because they create one-to-many or
many-to-many pairings. Missing keys and keys present on only one side are
skipped.

Both modes use the entire loaded datasets, including unsaved SpecID edits;
table filters do not restrict calculation. Choose **Include matched data** to
make the result self-contained, or **Lightweight result** to save only indices,
scores, and calculation metadata. Embedded results store each unique matched
record once and refer to it by number from every result row, so one spectrum
matching many candidates does not duplicate its pandas metadata or NumPy peak
arrays.

The `.mssim` file opens in a dedicated **Similarity Viewer**:

- Embedded results show **Open spectra** on each row to open the saved query
  and reference records as a mirrored spectrum comparison.
- **Filter**: combine numeric comparisons and key/text conditions.
- Click a column heading to sort; use Previous/Next for paging.
- The histogram and count/mean/Q1/median/Q3/min/max summarize all filtered rows,
  not just the current page. Choose 1–200 bins; click a bar or use the accessible
  frequency table to filter a similarity range. The final bin includes 1.0.
- Switch between a histogram and a box plot. Histogram counts support linear and
  log10 scales.
- **Save PNG…** opens an image preview where width, height, grid visibility,
  histogram color, box color, and count scale can be changed before saving.
- **Export…**: save the filtered/sorted results as `.mssim`, TSV, CSV, or Parquet.
  Only `.mssim` preserves calculation metadata, embedded matched data, and the
  export filter definition. Unreferenced embedded records are removed.
- **Reload**: reread the result from disk, retaining active filters. It does not
  recalculate spectra. Start another calculation from the dataset viewer to do so.
- Expand **Calculation metadata** to inspect parameters, creation time, input
  paths, dataset descriptions/attributes/tags, and row counts.

The result opens independently of the source files. The configured Python
environment must contain the version of `msentity` from this checkout, including
`msentity.similarity` (for development: `python -m pip install -e .`
from the repository root).

### Similarity file format (version 2)

`.mssim` is an HDF5 container with root attributes `format = msentity.similarity`
and `schema_version = 2`. `table.parquet` is a uint8 dataset containing the bytes
produced by pandas `DataFrame.to_parquet(index=False)`; `metadata.json` is a UTF-8
JSON scalar containing calculation/source metadata. Self-contained files store
the two compact match datasets and original source-index arrays under
`matched_data/1` and `matched_data/2`. Version 1 table-only files remain
readable. Writes replace the target atomically after the entire container has
been written.

The same calculation is available from the CLI:

```console
msentity similarity-by-key first.msds second.msds --key1 SpecID --key2 SpecID --output result.mssim
msentity library-search query.msds reference.msp --threshold 0.8 --output matches.mssim
```

## Assign SpecID

Click **Assign SpecID…** and enter a prefix (or leave it empty). IDs start at 1
and are zero-padded to the largest number: 12 spectra with prefix `SP` receive
`SP01` through `SP12`. This assigns IDs to every spectrum in the active dataset
in its original order, regardless of filters, sorting, or the current page.
Existing SpecID values are replaced only after confirmation.

The table refreshes after assignment. Use **Export…** with the SpecID column
selected to save the changes. Export respects current filters and visible
columns; clear filters to export all spectra. **Reload** or closing the viewer
discards assignments that have not been exported.

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

Open an `.msds`, `.msp`, or `.mgf` file normally. For TSV or CSV, use **MS
Entity: Open as TSV** or **MS Entity: Open as CSV** so ordinary tabular files
remain associated with VS Code's normal text editor. Click a spectrum button in the table;
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
The metadata area below the plot shows separate **Upper Metadata** and **Lower
Metadata** panels during comparison, using the columns from each spectrum's
source dataset.
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

Use **Copy TSV** or **Copy CSV** above the peak table to copy its complete,
unrounded values. **Save…** first asks for TSV or CSV and then opens the file
save dialog. A single spectrum has `m/z` and `Intensity` columns. A comparison
has separate Upper and Lower columns; peaks within the current tolerance share
a row, while an unmatched peak leaves the other side empty. Ascending or
descending m/z order is preserved independently on both sides. Export always
includes the complete spectra, independent of the current plot zoom.

Opening normally detects MSDS, MSP, or MGF from the extension. TSV and CSV are
intentionally not registered as custom-editor file extensions. To open one as a spectrum table,
right-click a file in Explorer and choose **MS Entity: Open as MSDS**, **Open as
MSP**, **Open as MGF**, **Open as TSV**, or **Open as CSV**. The same commands are available in the Command
Palette. Dataset **Export...** first asks for MSDS, MSP, MGF, TSV, or CSV and then for the
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

## Similarity tests

From the repository root:

```console
python -m unittest discover -s tests/TestProcessing -p 'Test*.py'
python -m unittest discover -s vscode-extension/tests -p 'test_*.py'
```

For the parameter wizard and browser interaction test, install Playwright in a
separate test environment (or make it available through `NODE_PATH`), install its
Chromium browser, and run `node --test vscode-extension/tests/test_similarity_ui.js`.
`MSENTITY_PYTHON` selects the Python executable; `CHROMIUM_PATH` optionally selects
an existing Chromium executable. The test uses the actual Python result backend
and a mocked VS Code host, and writes a screenshot to the system temporary directory.
Run `WIZARD_ONLY=1 node vscode-extension/tests/test_similarity_ui.js` to check
the VS Code parameter flow without Playwright or Chromium.
