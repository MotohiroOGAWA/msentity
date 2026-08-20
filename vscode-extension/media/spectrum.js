(() => {
  const vscode = acquireVsCodeApi();
  const app = document.getElementById("spectrum-app");
  let state = null;

  const esc = (v) => String(v ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
  const display = (v) => v == null ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v);

  // Return human-friendly tick intervals from the 1, 2, 2.5, 5, 10 series.
  const niceStep = (span, targetTicks = 6) => {
    const raw = Math.max(Number.EPSILON, span / targetTicks);
    const power = 10 ** Math.floor(Math.log10(raw));
    const fraction = raw / power;
    const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 2.5 ? 2.5 : fraction <= 5 ? 5 : 10;
    return nice * power;
  };

  const tickDecimals = (step) => Math.max(0, -Math.floor(Math.log10(step) + 1e-12));

  const ticks = (start, end, targetTicks = 6) => {
    const step = niceStep(end - start, targetTicks);
    const first = Math.ceil((start - step * 1e-9) / step) * step;
    const values = [];
    for (let value = first; value <= end + step * 1e-9 && values.length < 100; value += step) {
      values.push(Math.abs(value) < step * 1e-9 ? 0 : value);
    }
    return { step, values };
  };

  const save = async (blob, filename) => {
    const bytes = Array.from(new Uint8Array(await blob.arrayBuffer()));
    vscode.postMessage({ type: "save-image", filename, bytes });
  };

  const exportSvg = (format, title) => {
    const source = document.getElementById("spectrum-svg");
    if (!source) return;
    const svg = source.cloneNode(true);
    svg.querySelector("#drag-box")?.remove();
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    svg.setAttribute("width", "900");
    svg.setAttribute("height", "540");

    const styles = getComputedStyle(document.documentElement);
    const color = (name, fallback) => styles.getPropertyValue(name).trim() || fallback;
    const css = `
      text { font: 13px system-ui, sans-serif; fill: ${color("--plot-text", "#667085")} }
      .grid { stroke: ${color("--grid", "#d0d5dd")}; stroke-width: 1; opacity: .72 }
      .peak { stroke: ${color("--peak", "#2563eb")}; stroke-width: 2.4 }
      .peak.selected { stroke: ${color("--peak-selected", "#dc2626")}; stroke-width: 4 }
      .axis { stroke: ${color("--axis", "#344054")}; stroke-width: 1.6 }
      .axis-title { font-weight: 650; fill: ${color("--axis", "#344054")} }
      .mz-title { font-style: italic }
    `;
    const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
    style.textContent = css;
    svg.prepend(style);
    const markup = new XMLSerializer().serializeToString(svg);
    const basename = String(title || "spectrum").replace(/[^\w.-]+/g, "_");

    if (format === "svg") {
      save(new Blob([markup], { type: "image/svg+xml;charset=utf-8" }), `${basename}.svg`);
      return;
    }

    const image = new Image();
    const url = URL.createObjectURL(new Blob([markup], { type: "image/svg+xml;charset=utf-8" }));
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = 1800;
      canvas.height = 1080;
      const context = canvas.getContext("2d");
      context.scale(2, 2);
      context.drawImage(image, 0, 0, 900, 540);
      URL.revokeObjectURL(url);
      canvas.toBlob((blob) => blob && save(blob, `${basename}.png`), "image/png");
    };
    image.src = url;
  };

  function renderSpectrum(payload) {
    const spectrum = payload?.spectrum ?? { mz: [], intensity: [] };
    const row = payload?.row ?? {};
    const columns = Array.isArray(payload?.columns) ? payload.columns.map(String) : Object.keys(row);
    const globalIndex = Number.isInteger(payload?.globalIndex) ? payload.globalIndex : 0;
    const title = String(payload?.title || `Spectrum ${globalIndex + 1}`);
    const mz = Array.isArray(spectrum.mz) ? spectrum.mz.map(Number) : [];
    const intensity = Array.isArray(spectrum.intensity) ? spectrum.intensity.map(Number) : [];

    state = {
      spectrum,
      row,
      columns,
      globalIndex,
      title,
      selectedPeak: null,
      sortKey: "mz",
      sortAscending: true,
      domain: null
    };

    const metadata = columns.map((c) => `<div><dt>${esc(c)}</dt><dd>${esc(display(row[c]))}</dd></div>`).join("");
    app.className = "";
    app.innerHTML = `
      <main class="spectrum-viewer">
        <header class="spectrum-header">
          <div>
            <div class="eyebrow">MSENTITY · SPECTRUM ${globalIndex + 1}</div>
            <h1>${esc(title)}</h1>
          </div>
          <div class="actions"><span>${mz.length} peaks</span><button id="save-png">Save PNG</button><button id="save-svg">Save SVG</button><button id="reset-zoom">Reset zoom</button></div>
        </header>
        <div class="spectrum-workspace">
          <section class="plot-panel" id="plot-root"></section>
          <aside class="peaks-panel">
            <h2>Peaks</h2>
            <div class="peak-scroll">
              <table class="peak-table">
                <thead><tr><th><button class="sort-button" id="sort-mz">m/z ↑</button></th><th><button class="sort-button" id="sort-intensity">Intensity</button></th></tr></thead>
                <tbody id="peak-body"></tbody>
              </table>
            </div>
          </aside>
        </div>
        <section class="metadata-panel"><h2>Metadata</h2><dl>${metadata}</dl></section>
      </main>`;
    setupSpectrumPlot(mz, intensity);
  }

  function setupSpectrumPlot(mz, intensities) {
    if (!state) return;
    const peaks = mz.map((x, index) => ({ x, y: intensities[index] ?? 0, index }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    const root = document.getElementById("plot-root");
    const tbody = document.getElementById("peak-body");
    const reset = document.getElementById("reset-zoom");
    const savePng = document.getElementById("save-png");
    const saveSvg = document.getElementById("save-svg");
    const mzSort = document.getElementById("sort-mz");
    const intensitySort = document.getElementById("sort-intensity");
    if (!root || !tbody || !reset || !savePng || !saveSvg || !mzSort || !intensitySort) return;

    const W = 900, H = 540, L = 72, R = 32, T = 30, B = 62;
    const rawMax = peaks.length ? Math.max(...peaks.map((p) => p.x)) : 1;
    const fullXStep = niceStep(Math.max(1, rawMax), 6);
    const full = [0, Math.max(fullXStep, Math.ceil(rawMax / fullXStep) * fullXStep)];
    if (!Array.isArray(state.domain)) state.domain = [...full];
    let start = null, current = null, pressedPeak = null;

    const choose = (index) => {
      state.selectedPeak = index;
      draw();
      document.querySelector(`[data-peak-row="${index}"]`)?.scrollIntoView({ block: "nearest" });
    };

    const draw = () => {
      if (!state) return;
      const domain = state.domain;
      const domainWidth = Math.max(Number.EPSILON, domain[1] - domain[0]);
      const shown = peaks.filter((p) => p.x >= domain[0] && p.x <= domain[1]);
      const rawYMax = Math.max(1, ...shown.map((p) => Math.max(0, p.y)));
      const yStep = niceStep(rawYMax, 6);
      const yMax = Math.max(yStep, Math.ceil(rawYMax / yStep) * yStep);
      const sx = (x) => L + (x - domain[0]) / domainWidth * (W - L - R);
      const sy = (y) => H - B - Math.max(0, y) / yMax * (H - T - B);

      let grid = "";
      const xTicks = ticks(domain[0], domain[1]);
      for (const value of xTicks.values) {
        const x = sx(value);
        grid += `<line x1="${x}" y1="${T}" x2="${x}" y2="${H - B}" class="grid"/>`;
        grid += `<text x="${x}" y="${H - B + 26}" text-anchor="middle">${value.toFixed(tickDecimals(xTicks.step))}</text>`;
      }
      const yTicks = ticks(0, yMax);
      for (const value of yTicks.values) {
        const y = sy(value);
        grid += `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" class="grid"/>`;
        grid += `<text x="${L - 10}" y="${y + 4}" text-anchor="end">${value.toFixed(tickDecimals(yTicks.step))}</text>`;
      }
      const sticks = shown.map((p) => `<line data-peak="${p.index}" x1="${sx(p.x)}" y1="${H - B}" x2="${sx(p.x)}" y2="${sy(p.y)}" class="peak ${state.selectedPeak === p.index ? "selected" : ""}"><title>m/z ${p.x} · intensity ${p.y}</title></line>`).join("");
      root.innerHTML = `<svg id="spectrum-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Mass spectrum"><g class="spectrum-grid">${grid}</g><line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" class="axis"/><line x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" class="axis"/>${sticks}<rect id="drag-box" x="${L}" y="${T}" width="0" height="${H - T - B}" class="selection"/><text x="${(L + W - R) / 2}" y="${H - 12}" text-anchor="middle" class="axis-title mz-title">m/z</text><text transform="translate(17 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle" class="axis-title">Intensity</text></svg>`;

      const ordered = [...peaks].sort((a, b) => (state.sortKey === "mz" ? a.x - b.x : a.y - b.y) * (state.sortAscending ? 1 : -1));
      tbody.innerHTML = ordered.map((p) => `<tr data-peak-row="${p.index}" class="peak-row ${state.selectedPeak === p.index ? "selected" : ""}"><td><button data-peak-button="${p.index}">${p.x.toFixed(5)}</button></td><td><button data-peak-button="${p.index}">${p.y.toLocaleString()}</button></td></tr>`).join("");
      mzSort.textContent = `m/z ${state.sortKey === "mz" ? (state.sortAscending ? "↑" : "↓") : ""}`;
      intensitySort.textContent = `Intensity ${state.sortKey === "intensity" ? (state.sortAscending ? "↑" : "↓") : ""}`;

      root.querySelectorAll("[data-peak]").forEach((el) => el.addEventListener("click", (e) => {
        e.stopPropagation();
        choose(Number(el.dataset.peak));
      }));
      tbody.querySelectorAll("[data-peak-button]").forEach((el) => el.addEventListener("click", () => choose(Number(el.dataset.peakButton))));

      const svg = root.querySelector("#spectrum-svg");
      if (!svg) return;
      const pointerX = (e) => {
        const rect = svg.getBoundingClientRect();
        return Math.min(W - R, Math.max(L, (e.clientX - rect.left) / rect.width * W));
      };
      svg.addEventListener("pointerdown", (e) => {
        const peak = e.target.closest?.("[data-peak]");
        pressedPeak = peak ? Number(peak.dataset.peak) : null;
        start = pointerX(e);
        current = start;
        svg.setPointerCapture(e.pointerId);
      });
      svg.addEventListener("pointermove", (e) => {
        if (start === null) return;
        current = pointerX(e);
        const box = svg.querySelector("#drag-box");
        if (box) {
          box.setAttribute("x", String(Math.min(start, current)));
          box.setAttribute("width", String(Math.abs(current - start)));
        }
      });
      svg.addEventListener("pointerup", () => {
        const dragged = start !== null && current !== null && Math.abs(current - start) > 8;
        if (dragged) {
          const toMz = (x) => domain[0] + (x - L) / (W - L - R) * domainWidth;
          state.domain = [toMz(Math.min(start, current)), toMz(Math.max(start, current))];
        }
        const clickedPeak = pressedPeak;
        start = current = pressedPeak = null;
        if (dragged) draw();
        else if (clickedPeak !== null) choose(clickedPeak);
        else svg.querySelector("#drag-box")?.setAttribute("width", "0");
      });
    };

    const changeSort = (key) => {
      if (!state) return;
      if (state.sortKey === key) state.sortAscending = !state.sortAscending;
      else {
        state.sortKey = key;
        state.sortAscending = true;
      }
      draw();
    };
    mzSort.onclick = () => changeSort("mz");
    intensitySort.onclick = () => changeSort("intensity");
    reset.onclick = () => {
      state.domain = [...full];
      state.selectedPeak = null;
      draw();
    };
    savePng.onclick = () => exportSvg("png", state?.title);
    saveSvg.onclick = () => exportSvg("svg", state?.title);
    draw();
  }

  window.addEventListener("message", (event) => {
    if (event.data?.type === "spectrum") renderSpectrum(event.data.payload);
  });

  vscode.postMessage({ type: "ready" });
})();
