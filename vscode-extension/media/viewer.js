(() => {
  const vscode = acquireVsCodeApi();
  const app = document.getElementById("app");
  const filename = app.dataset.filename || "dataset";

  let value = null;
  let selectedColumns = [];
  let knownColumnsKey = "";
  let loadingPage = false;
  let columnMenuOpen = false;
  let columnMenuScrollTop = 0;
  let filterMenuOpen = false;
  let filters = [];
  let rowSort = [];
  let filterSequence = 0;
  let selectedSpectrumIndex = null;
  let loadingProgress = null;
  let exporting = false;
  let datasetOptions = [];
  let activeDatasetId = "";
  const datasetPages = new Map();

  const esc = (v) => String(v ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
  const display = (v) => v == null ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v);
  const columns = () => Array.isArray(value?.all_columns) ? value.all_columns.map(String) : Array.isArray(value?.columns) ? value.columns.map(String) : [];
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

  function viewRequest(pageNumber = 0) {
    return { type: "page-request", page: pageNumber, datasetId: activeDatasetId, filters, sort: rowSort, columns: selectedColumns };
  }

  function refreshView() {
    loadingPage = true;
    selectedSpectrumIndex = null;
    render();
    vscode.postMessage(viewRequest(0));
  }

  function requestPage(requested) {
    const target = Math.min(pageCount() - 1, Math.max(0, Number(requested) || 0));
    if (target === page() || loadingPage) return;
    loadingPage = true;
    selectedSpectrumIndex = null;
    render();
    vscode.postMessage(viewRequest(target));
  }

  function toggleColumn(column) {
    columnMenuScrollTop = document.getElementById("column-menu")?.scrollTop || 0;
    selectedColumns = selectedColumns.includes(column)
      ? selectedColumns.filter((x) => x !== column)
      : columns().filter((x) => x === column || selectedColumns.includes(x));
    render();
  }

  function moveColumn(column, offset) {
    columnMenuScrollTop = document.getElementById("column-menu")?.scrollTop || 0;
    const index = selectedColumns.indexOf(column);
    const target = index + offset;
    if (index < 0 || target < 0 || target >= selectedColumns.length) return;
    [selectedColumns[index], selectedColumns[target]] = [selectedColumns[target], selectedColumns[index]];
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
    const pageRows = rows().map((row, index) => ({ row, index }));
    const cols = columns();
    const allSelected = cols.length > 0 && selectedColumns.length === cols.length;
    const start = rows().length ? rowOffset() + 1 : 0;
    const end = rowOffset() + rows().length;
    const sortFor = (column) => {
      const index = rowSort.findIndex((item) => item.column === column);
      return index < 0 ? null : { ...rowSort[index], priority: index + 1 };
    };

    const columnMenu = columnMenuOpen ? `
      <div class="column-menu" id="column-menu">
        <button class="column-option" data-action="all-columns"><span class="checkmark ${allSelected ? "checked" : ""}">✓</span><strong>All columns</strong></button>
        ${cols.map((c) => `<div class="column-option"><button class="column-toggle" data-column="${esc(c)}"><span class="checkmark ${selectedColumns.includes(c) ? "checked" : ""}">✓</span><span>${esc(c)}</span></button>${selectedColumns.includes(c) ? `<span class="column-order"><button data-move-column="${esc(c)}" data-offset="-1" title="Move left" ${selectedColumns.indexOf(c) === 0 ? "disabled" : ""}>↑</button><button data-move-column="${esc(c)}" data-offset="1" title="Move right" ${selectedColumns.indexOf(c) === selectedColumns.length - 1 ? "disabled" : ""}>↓</button></span>` : ""}</div>`).join("")}
      </div>` : "";

    const filterMenu = filterMenuOpen ? `
      <div class="filter-menu" id="filter-menu">
        <div class="filter-title"><strong>Filters</strong><button id="add-filter" title="Add filter">+</button></div>
        ${filters.length ? filters.map((filter) => `<div class="filter-row" data-filter-id="${filter.id}">
          <select data-filter-field="column" aria-label="Column">${cols.map((c) => `<option value="${esc(c)}" ${c === filter.column ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>
          <select data-filter-field="operator" aria-label="Operator">${[
            ["text_eq", "= (text)"], ["numeric_eq", "= (number)"], ["!=", "!= (text)"], ["contains", "contains"],
            [">", "> (number)"], [">=", ">= (number)"], ["<", "< (number)"], ["<=", "<= (number)"]
          ].map(([op, label]) => `<option value="${esc(op)}" ${op === filter.operator ? "selected" : ""}>${esc(label)}</option>`).join("")}</select>
          <input data-filter-field="value" value="${esc(filter.value)}" placeholder="Value" />
          <button class="remove-filter" title="Remove filter">×</button>
        </div>`).join("") : `<div class="filter-empty">Press + to add a condition.</div>`}
        <div class="filter-actions"><button id="clear-filters" class="secondary-button" ${filters.length ? "" : "disabled"}>Clear</button><button id="apply-filters">Apply</button></div>
      </div>` : "";

    app.innerHTML = `
      <div class="viewer">
        <div class="toolbar">
          <div><div class="dataset-heading"><select id="dataset-select" aria-label="Active dataset">${datasetOptions.map((dataset) => `<option value="${esc(dataset.id)}" ${dataset.id === activeDatasetId ? "selected" : ""}>${esc(dataset.name)}</option>`).join("")}</select><button class="secondary-button" id="add-dataset">Add dataset…</button></div><div class="summary">${totalRows().toLocaleString()} spectra · ${selectedColumns.length}/${cols.length} columns${value.description ? ` · ${esc(value.description)}` : ""}</div></div>
          <div class="tools">
            <div class="columns"><button id="columns-button">Columns</button>${columnMenu}</div>
            <div class="filters"><button id="filter-button" class="${filters.length ? "active" : ""}">Filter${filters.length ? ` (${filters.length})` : ""}</button>${filterMenu}</div>
            <button class="secondary-button" id="export-button" ${exporting ? "disabled" : ""}>${exporting ? "Exporting…" : "Export…"}</button>
            <button class="secondary-button" id="reload-button" title="Reload dataset from disk">Reload</button>
          </div>
        </div>
        <div class="table-wrap"><table class="dataset-table"><thead><tr><th class="row-column">Row</th><th class="spectrum-column">Spectrum</th>${selectedColumns.map((c) => { const item = sortFor(c); return `<th><button class="table-sort" data-sort-column="${esc(c)}" title="Click: ascending → descending → remove sort">${esc(c)}<span>${item ? `${item.priority}${item.direction === "asc" ? "▲" : "▼"}` : ""}</span></button></th>`; }).join("")}</tr></thead><tbody>
          ${pageRows.length ? pageRows.map(({ row, index }) => `<tr class="${selectedSpectrumIndex === index ? "selected-record" : ""}"><td class="row-column">${rowOffset() + index + 1}</td><td class="spectrum-column"><button class="spectrum-button" data-spectrum-index="${index}" title="Show spectrum ${rowOffset() + index + 1}"><svg viewBox="0 0 28 22"><path d="M2 19h24M4 18V13m4 5V7m4 11v-4m4 4V3m4 15V9m4 9v-7"/></svg></button></td>${selectedColumns.map((c) => `<td title="${esc(display(row?.[c]))}">${esc(display(row?.[c]))}</td>`).join("")}</tr>`).join("") : `<tr><td colspan="${selectedColumns.length + 2}" class="empty">No spectra match the filters.</td></tr>`}
        </tbody></table></div>
        <div class="pagination"><span>${start}–${end} of ${totalRows()}</span><div class="page-controls"><button id="prev-page" ${page() === 0 || loadingPage ? "disabled" : ""}>‹</button><span><input id="page-input" type="number" min="1" max="${pageCount()}" value="${page() + 1}" ${loadingPage ? "disabled" : ""}/> / ${pageCount()}</span><button id="next-page" ${page() >= pageCount() - 1 || loadingPage ? "disabled" : ""}>›</button></div><span>${loadingPage ? "Loading…" : `${pageSize()} rows/page`}</span></div>
      </div>`;

    const restoredColumnMenu = document.getElementById("column-menu");
    if (restoredColumnMenu) restoredColumnMenu.scrollTop = columnMenuScrollTop;

    document.getElementById("columns-button")?.addEventListener("click", (e) => { e.stopPropagation(); columnMenuOpen = !columnMenuOpen; render(); });
    document.querySelector("[data-action='all-columns']")?.addEventListener("click", () => { columnMenuScrollTop = document.getElementById("column-menu")?.scrollTop || 0; selectedColumns = allSelected ? [] : [...cols]; render(); });
    document.querySelectorAll("[data-column]").forEach((el) => el.addEventListener("click", () => toggleColumn(el.dataset.column)));
    document.querySelectorAll("[data-move-column]").forEach((el) => el.addEventListener("click", () => moveColumn(el.dataset.moveColumn, Number(el.dataset.offset))));
    document.getElementById("filter-button")?.addEventListener("click", (e) => { e.stopPropagation(); filterMenuOpen = !filterMenuOpen; columnMenuOpen = false; render(); });
    document.getElementById("add-filter")?.addEventListener("click", () => { filters.push({ id: ++filterSequence, column: cols[0] || "", operator: "text_eq", value: "" }); render(); });
    document.querySelectorAll("[data-filter-id]").forEach((row) => {
      const filter = filters.find((item) => item.id === Number(row.dataset.filterId));
      row.querySelectorAll("[data-filter-field]").forEach((control) => control.addEventListener("input", () => { filter[control.dataset.filterField] = control.value; }));
      row.querySelector(".remove-filter")?.addEventListener("click", () => { filters = filters.filter((item) => item !== filter); render(); });
    });
    document.getElementById("clear-filters")?.addEventListener("click", () => { filters = []; refreshView(); });
    document.getElementById("apply-filters")?.addEventListener("click", refreshView);
    document.querySelectorAll("[data-sort-column]").forEach((el) => el.addEventListener("click", () => {
      const column = el.dataset.sortColumn;
      const index = rowSort.findIndex((item) => item.column === column);
      if (index < 0) rowSort.push({ column, direction: "asc" });
      else if (rowSort[index].direction === "asc") rowSort[index] = { column, direction: "desc" };
      else rowSort.splice(index, 1);
      refreshView();
    }));
    document.getElementById("dataset-select")?.addEventListener("change", (event) => {
      activeDatasetId = event.currentTarget.value;
      const savedPage = datasetPages.get(activeDatasetId) ?? 0;
      loadingPage = true;
      selectedSpectrumIndex = null;
      filters = [];
      rowSort = [];
      selectedColumns = [];
      knownColumnsKey = "";
      render();
      vscode.postMessage(viewRequest(savedPage));
    });
    document.getElementById("add-dataset")?.addEventListener("click", () => vscode.postMessage({ type: "add-dataset" }));
    document.getElementById("export-button")?.addEventListener("click", () => { exporting = true; render(); vscode.postMessage({ type: "export-dataset", datasetId: activeDatasetId, datasetPath: value.dataset_path, filters, sort: rowSort, columns: selectedColumns }); });
    document.getElementById("reload-button")?.addEventListener("click", () => { loadingPage = true; selectedSpectrumIndex = null; render(); vscode.postMessage({ type: "reload", datasetId: activeDatasetId, filters, sort: rowSort }); });
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
        datasetName: value.dataset_name || filename,
        datasetPath: value.dataset_path || ""
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
        datasetPages.set(activeDatasetId, 0);
      }
      vscode.postMessage({ type: "ready", datasetId: activeDatasetId });
    } else if (message?.type === "dataset-added") {
      const dataset = message.dataset;
      if (dataset && !datasetOptions.some((item) => item.id === dataset.id)) datasetOptions.push(dataset);
      if (dataset) {
        activeDatasetId = dataset.id;
        filters = [];
        rowSort = [];
        selectedColumns = [];
        knownColumnsKey = "";
        if (!datasetPages.has(activeDatasetId)) datasetPages.set(activeDatasetId, 0);
      }
    } else if (message?.type === "loading-progress") {
      loadingProgress = message;
      renderLoading(`Reading ${String(message.file_type || "dataset").toUpperCase()}…`);
    } else if (message?.type === "dataset-page") {
      value = message.value;
      activeDatasetId = String(value?.dataset_id || activeDatasetId);
      datasetPages.set(activeDatasetId, Number.isInteger(value?.page) ? value.page : 0);
      loadingProgress = null;
      loadingPage = false;
      columnMenuOpen = false;
      filterMenuOpen = false;
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
    else if (filterMenuOpen && !e.target.closest?.(".filters")) { filterMenuOpen = false; render(); }
  });

  renderLoading();
})();
