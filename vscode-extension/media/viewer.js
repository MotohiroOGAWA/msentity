(() => {
  const vscode = acquireVsCodeApi();
  const app = document.getElementById("app");
  const filename = app.dataset.filename || "dataset";

  let value = null;
  let query = "";
  let selectedColumns = [];
  let knownColumnsKey = "";
  let loadingPage = false;
  let columnMenuOpen = false;
  let selectedSpectrumIndex = null;
  let loadingProgress = null;
  let exporting = false;
  let datasetOptions = [];
  let activeDatasetId = "";

  const esc = (v) => String(v ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
  const display = (v) => v == null ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v);
  const columns = () => Array.isArray(value?.columns) ? value.columns.map(String) : [];
  const rows = () => Array.isArray(value?.rows) ? value.rows : [];
  const spectra = () => Array.isArray(value?.spectra) ? value.spectra : [];
  const page = () => Number.isInteger(value?.page) ? value.page : 0;
  const pageSize = () => Number.isInteger(value?.page_size) ? value.page_size : 20;
  const rowOffset = () => Number.isInteger(value?.row_offset) ? value.row_offset : page() * pageSize();
  const totalRows = () => Number.isInteger(value?.total_rows) ? value.total_rows : rows().length;
  const pageCount = () => Number.isInteger(value?.total_pages) ? value.total_pages : Math.max(1, Math.ceil(totalRows() / pageSize()));

  function ensureColumns() {
    const cols = columns();
    const key = JSON.stringify(cols);
    if (key !== knownColumnsKey) {
      knownColumnsKey = key;
      selectedColumns = [...cols];
    }
  }

  function titleFor(row, index) {
    return String(row?.Name ?? row?.name ?? row?.TITLE ?? row?.Title ?? row?.ID ?? row?.id ?? `Spectrum ${index + 1}`);
  }

  function filteredRows() {
    const normalized = query.trim().toLocaleLowerCase();
    return rows().map((row, index) => ({ row, index })).filter(({ row }) =>
      !normalized || columns().some((c) => String(row?.[c] ?? "").toLocaleLowerCase().includes(normalized))
    );
  }

  function requestPage(requested) {
    const target = Math.min(pageCount() - 1, Math.max(0, Number(requested) || 0));
    if (target === page() || loadingPage) return;
    loadingPage = true;
    selectedSpectrumIndex = null;
    render();
    vscode.postMessage({ type: "page-request", page: target, datasetId: activeDatasetId });
  }

  function toggleColumn(column) {
    selectedColumns = selectedColumns.includes(column)
      ? selectedColumns.filter((x) => x !== column)
      : columns().filter((x) => x === column || selectedColumns.includes(x));
    render();
  }

  function renderLoading(text = "Loading msentity dataset…") {
    const progress = loadingProgress;
    const percent = Math.max(0, Math.min(100, Number(progress?.percent) || 0));
    const details = progress ? `${percent.toFixed(1)}% · ${Number(progress.processed || 0).toLocaleString()} / ${Number(progress.total || 0).toLocaleString()} bytes · ${Number(progress.success || 0).toLocaleString()} spectra` : "Starting Python backend…";
    app.innerHTML = `<div class="status"><div class="loading-card"><div class="loading-title">${esc(text)}</div><progress max="100" value="${percent}"></progress><div class="loading-details">${esc(details)}</div></div></div>`;
  }

  function renderError(title, message) {
    app.innerHTML = `<div class="status"><div class="error-card"><h2>${esc(title)}</h2><pre>${esc(message)}</pre><button class="secondary-button" id="show-logs">Show logs</button></div></div>`;
    document.getElementById("show-logs")?.addEventListener("click", () => vscode.postMessage({ type: "show-logs" }));
  }

  function render() {
    if (!value) return renderLoading();
    ensureColumns();
    const pageRows = filteredRows();
    const cols = columns();
    const allSelected = cols.length > 0 && selectedColumns.length === cols.length;
    const start = rows().length ? rowOffset() + 1 : 0;
    const end = rowOffset() + rows().length;

    const columnMenu = columnMenuOpen ? `
      <div class="column-menu" id="column-menu">
        <button class="column-option" data-action="all-columns"><span class="checkmark ${allSelected ? "checked" : ""}">✓</span><strong>All columns</strong></button>
        ${cols.map((c) => `<button class="column-option" data-column="${esc(c)}"><span class="checkmark ${selectedColumns.includes(c) ? "checked" : ""}">✓</span><span>${esc(c)}</span></button>`).join("")}
      </div>` : "";

    app.innerHTML = `
      <div class="viewer">
        <div class="toolbar">
          <div><div class="dataset-heading"><select id="dataset-select" aria-label="Active dataset">${datasetOptions.map((dataset) => `<option value="${esc(dataset.id)}" ${dataset.id === activeDatasetId ? "selected" : ""}>${esc(dataset.name)}</option>`).join("")}</select><button class="secondary-button" id="add-dataset">Add dataset…</button></div><div class="summary">${totalRows().toLocaleString()} spectra · ${selectedColumns.length}/${cols.length} columns${value.description ? ` · ${esc(value.description)}` : ""}</div></div>
          <div class="tools">
            <div class="columns"><button id="columns-button">Columns</button>${columnMenu}</div>
            <label class="search"><input id="search-input" value="${esc(query)}" placeholder="Filter current page…" /></label>
            <button class="secondary-button" id="export-button" ${exporting ? "disabled" : ""}>${exporting ? "Exporting…" : "Export…"}</button>
            <button class="secondary-button" id="reload-button" title="Reload dataset from disk">Reload</button>
          </div>
        </div>
        <div class="table-wrap"><table class="dataset-table"><thead><tr><th class="row-column">Row</th><th class="spectrum-column">Spectrum</th>${selectedColumns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>
          ${pageRows.length ? pageRows.map(({ row, index }) => `<tr class="${selectedSpectrumIndex === index ? "selected-record" : ""}"><td class="row-column">${rowOffset() + index + 1}</td><td class="spectrum-column"><button class="spectrum-button" data-spectrum-index="${index}" title="Show spectrum ${rowOffset() + index + 1}"><svg viewBox="0 0 28 22"><path d="M2 19h24M4 18V13m4 5V7m4 11v-4m4 4V3m4 15V9m4 9v-7"/></svg></button></td>${selectedColumns.map((c) => `<td title="${esc(display(row?.[c]))}">${esc(display(row?.[c]))}</td>`).join("")}</tr>`).join("") : `<tr><td colspan="${selectedColumns.length + 2}" class="empty">No spectra match this search.</td></tr>`}
        </tbody></table></div>
        <div class="pagination"><span>${start}–${end} of ${totalRows()}</span><div class="page-controls"><button id="prev-page" ${page() === 0 || loadingPage ? "disabled" : ""}>‹</button><span><input id="page-input" type="number" min="1" max="${pageCount()}" value="${page() + 1}" ${loadingPage ? "disabled" : ""}/> / ${pageCount()}</span><button id="next-page" ${page() >= pageCount() - 1 || loadingPage ? "disabled" : ""}>›</button></div><span>${loadingPage ? "Loading…" : `${pageSize()} rows/page`}</span></div>
      </div>`;

    document.getElementById("columns-button")?.addEventListener("click", (e) => { e.stopPropagation(); columnMenuOpen = !columnMenuOpen; render(); });
    document.querySelector("[data-action='all-columns']")?.addEventListener("click", () => { selectedColumns = allSelected ? [] : [...cols]; render(); });
    document.querySelectorAll("[data-column]").forEach((el) => el.addEventListener("click", () => toggleColumn(el.dataset.column)));
    document.getElementById("search-input")?.addEventListener("input", (e) => { query = e.currentTarget.value; render(); });
    document.getElementById("dataset-select")?.addEventListener("change", (event) => {
      activeDatasetId = event.currentTarget.value;
      loadingPage = true;
      selectedSpectrumIndex = null;
      render();
      vscode.postMessage({ type: "page-request", page: 0, datasetId: activeDatasetId });
    });
    document.getElementById("add-dataset")?.addEventListener("click", () => vscode.postMessage({ type: "add-dataset" }));
    document.getElementById("export-button")?.addEventListener("click", () => { exporting = true; render(); vscode.postMessage({ type: "export-dataset", datasetId: activeDatasetId, datasetPath: value.dataset_path }); });
    document.getElementById("reload-button")?.addEventListener("click", () => { loadingPage = true; selectedSpectrumIndex = null; render(); vscode.postMessage({ type: "reload", datasetId: activeDatasetId }); });
    document.getElementById("prev-page")?.addEventListener("click", () => requestPage(page() - 1));
    document.getElementById("next-page")?.addEventListener("click", () => requestPage(page() + 1));
    const pageInput = document.getElementById("page-input");
    const go = () => { const n = Number.parseInt(pageInput.value, 10); if (Number.isFinite(n)) requestPage(n - 1); };
    pageInput?.addEventListener("change", go);
    pageInput?.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    document.querySelectorAll("[data-spectrum-index]").forEach((el) => el.addEventListener("click", () => openSpectrum(Number(el.dataset.spectrumIndex))));
  }

  function openSpectrum(index) {
    const spectrum = spectra()[index] ?? { mz: [], intensity: [] };
    const row = rows()[index] ?? {};
    const globalIndex = rowOffset() + index;
    selectedSpectrumIndex = index;
    render();
    vscode.postMessage({
      type: "open-spectrum",
      payload: {
        spectrum,
        row,
        columns: columns(),
        globalIndex,
        title: titleFor(row, globalIndex),
        datasetId: activeDatasetId,
        datasetName: value.dataset_name || filename
      }
    });
  }

  window.addEventListener("message", (event) => {
    const message = event.data;
    if (message?.type === "backend-ready") {
      loadingProgress = null;
      if (message.dataset) {
        datasetOptions = [message.dataset];
        activeDatasetId = message.dataset.id;
      }
      vscode.postMessage({ type: "ready", datasetId: activeDatasetId });
    } else if (message?.type === "dataset-added") {
      const dataset = message.dataset;
      if (dataset && !datasetOptions.some((item) => item.id === dataset.id)) datasetOptions.push(dataset);
      if (dataset) activeDatasetId = dataset.id;
    } else if (message?.type === "loading-progress") {
      loadingProgress = message;
      renderLoading(`Reading ${String(message.file_type || "dataset").toUpperCase()}…`);
    } else if (message?.type === "dataset-page") {
      value = message.value;
      activeDatasetId = String(value?.dataset_id || activeDatasetId);
      loadingProgress = null;
      loadingPage = false;
      query = "";
      columnMenuOpen = false;
      selectedSpectrumIndex = null;
      render();
    } else if (message?.type === "export-start") {
      exporting = true;
      render();
    } else if (message?.type === "export-complete") {
      exporting = false;
      render();
      vscode.postMessage({ type: "export-notification", path: message.path, totalRows: message.total_rows });
    } else if (message?.type === "export-cancelled") {
      exporting = false;
      render();
    } else if (message?.type === "export-error") {
      exporting = false;
      render();
      vscode.postMessage({ type: "export-error-notification", message: message.message });
    } else if (message?.type === "error") {
      exporting = false;
      renderError(message.title || "Error", message.message || "Unknown error");
    }
  });

  document.addEventListener("click", (e) => {
    if (columnMenuOpen && !e.target.closest?.(".columns")) { columnMenuOpen = false; render(); }
  });

  renderLoading();
})();
