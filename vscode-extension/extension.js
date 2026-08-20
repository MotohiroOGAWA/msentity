const vscode = require("vscode");
const path = require("path");
const { spawn } = require("child_process");
const crypto = require("crypto");

const VIEW_TYPE = "msentity.spectrumViewer";
let outputChannel;

class MSEntityDocument {
  constructor(uri) {
    this.uri = uri;
  }
  dispose() {}
}

class MSEntityViewerProvider {
  constructor(context) {
    this.context = context;
    this.spectrumPanels = new Map();
  }

  async openCustomDocument(uri) {
    return new MSEntityDocument(uri);
  }

  async resolveCustomEditor(document, webviewPanel) {
    const webview = webviewPanel.webview;
    webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, "media")]
    };
    webview.html = getDatasetWebviewHtml(webview, this.context.extensionUri, document.uri);

    const config = vscode.workspace.getConfiguration("msentitySpectrumViewer");
    const pythonPath = String(config.get("pythonPath", "python"));
    const pageSize = Math.max(1, Math.min(500, Number(config.get("pageSize", 20)) || 20));
    const backendPath = this.context.asAbsolutePath(path.join("python", "backend.py"));

    outputChannel.appendLine(`[open] ${document.uri.fsPath}`);
    outputChannel.appendLine(`[python] ${pythonPath}`);

    const child = spawn(
      pythonPath,
      ["-u", backendPath, document.uri.fsPath, "--page-size", String(pageSize)],
      {
        cwd: path.dirname(document.uri.fsPath),
        env: process.env,
        stdio: ["pipe", "pipe", "pipe"]
      }
    );

    let disposed = false;
    let stdoutBuffer = "";
    const send = (message) => {
      if (!disposed) webview.postMessage(message);
    };

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdoutBuffer += chunk;
      const lines = stdoutBuffer.split(/\r?\n/);
      stdoutBuffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        if (!line.startsWith("MSENTITY_JSON:")) {
          outputChannel.appendLine(`[backend stdout] ${line}`);
          continue;
        }
        try {
          send(JSON.parse(line.slice("MSENTITY_JSON:".length)));
        } catch (error) {
          outputChannel.appendLine(`[protocol error] ${String(error)} :: ${line}`);
        }
      }
    });

    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      for (const line of chunk.split(/\r?\n/)) {
        if (line.trim()) outputChannel.appendLine(`[backend stderr] ${line}`);
      }
    });

    child.on("error", (error) => {
      outputChannel.appendLine(`[spawn error] ${String(error)}`);
      send({
        type: "error",
        title: "Could not start Python",
        message: `${error.message}\n\nSet msentitySpectrumViewer.pythonPath to the Python environment where msentity is installed.`
      });
    });

    child.on("exit", (code, signal) => {
      outputChannel.appendLine(`[backend exit] code=${code} signal=${signal}`);
      if (!disposed && code !== 0) {
        send({
          type: "error",
          title: "msentity backend stopped",
          message: `The Python backend exited with code ${code ?? "unknown"}. Open “MS Entity: Show Spectrum Viewer Logs” for details.`
        });
      }
    });

    const writeRequest = (request) => {
      if (child.exitCode !== null || !child.stdin.writable) return;
      child.stdin.write(`${JSON.stringify(request)}\n`);
    };

    const messageDisposable = webview.onDidReceiveMessage((message) => {
      switch (message?.type) {
        case "ready":
          writeRequest({ type: "page", page: 0 });
          break;
        case "page-request":
          writeRequest({ type: "page", page: Number(message.page) || 0 });
          break;
        case "reload":
          writeRequest({ type: "reload" });
          break;
        case "open-spectrum":
          this.showSpectrum(document.uri, message.payload);
          break;
        case "show-logs":
          outputChannel.show(true);
          break;
      }
    });

    webviewPanel.onDidDispose(() => {
      disposed = true;
      messageDisposable.dispose();
      if (child.exitCode === null) child.kill();
    });
  }

  showSpectrum(documentUri, payload) {
    if (!payload || typeof payload !== "object") return;
    const key = documentUri.toString();
    let entry = this.spectrumPanels.get(key);

    if (!entry) {
      const panel = vscode.window.createWebviewPanel(
        "msentity.spectrumPlot",
        "Mass Spectrum",
        { viewColumn: vscode.ViewColumn.Beside, preserveFocus: false },
        {
          enableScripts: true,
          retainContextWhenHidden: true,
          localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, "media")]
        }
      );
      panel.iconPath = undefined;
      panel.webview.html = getSpectrumWebviewHtml(panel.webview, this.context.extensionUri);
      entry = { panel, ready: false, latest: payload };
      this.spectrumPanels.set(key, entry);

      panel.webview.onDidReceiveMessage((message) => {
        if (message?.type === "ready") {
          entry.ready = true;
          if (entry.latest) panel.webview.postMessage({ type: "spectrum", payload: entry.latest });
        } else if (message?.type === "save-image") {
          this.saveImage(documentUri, message);
        } else if (message?.type === "export-error") {
          vscode.window.showErrorMessage(String(message.message || "Could not export spectrum image."));
        }
      });
      panel.onDidDispose(() => this.spectrumPanels.delete(key));

      const config = vscode.workspace.getConfiguration("msentitySpectrumViewer");
      if (config.get("spectrumFloatingWindow", true)) {
        // VS Code 1.85+ supports floating editor windows. createWebviewPanel cannot
        // target one directly, so make the new spectrum editor active and move it.
        setTimeout(async () => {
          try {
            await vscode.commands.executeCommand("workbench.action.moveEditorToNewWindow");
          } catch (error) {
            outputChannel.appendLine(`[floating window] Could not move spectrum editor: ${String(error)}`);
          }
        }, 0);
      }
    } else {
      entry.latest = payload;
      // Reveal in its current location. Omitting a view column is important when
      // the panel lives in a floating VS Code window because specifying a column
      // would move it back into the main window.
      entry.panel.reveal(undefined, true);
    }

    entry.latest = payload;
    entry.panel.title = `${payload.title || `Spectrum ${(payload.globalIndex ?? 0) + 1}`} · Mass Spectrum`;
    if (entry.ready) entry.panel.webview.postMessage({ type: "spectrum", payload });
  }

  async saveImage(documentUri, message) {
    const filename = String(message?.filename || "spectrum.svg").replace(/[^\w.-]+/g, "_");
    const bytes = Array.isArray(message?.bytes) ? Uint8Array.from(message.bytes) : null;
    if (!bytes) return;
    const target = await vscode.window.showSaveDialog({
      defaultUri: vscode.Uri.joinPath(documentUri, "..", filename),
      filters: filename.endsWith(".png") ? { "PNG image": ["png"] } : { "SVG image": ["svg"] }
    });
    if (!target) return;
    await vscode.workspace.fs.writeFile(target, bytes);
    vscode.window.showInformationMessage(`Saved ${path.basename(target.fsPath)}`);
  }

  dispose() {
    for (const { panel } of this.spectrumPanels.values()) panel.dispose();
    this.spectrumPanels.clear();
  }
}

function getDatasetWebviewHtml(webview, extensionUri, documentUri) {
  const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, "media", "viewer.js"));
  const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, "media", "viewer.css"));
  const nonce = crypto.randomBytes(16).toString("base64");
  const filename = path.basename(documentUri.fsPath).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[c]));
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src blob:; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';" />
  <link rel="stylesheet" href="${styleUri}" />
  <title>${filename} · msentity</title>
</head>
<body>
  <div id="app" data-filename="${filename}"></div>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
}

function getSpectrumWebviewHtml(webview, extensionUri) {
  const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, "media", "spectrum.js"));
  const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, "media", "spectrum.css"));
  const nonce = crypto.randomBytes(16).toString("base64");
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src blob: data:; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';" />
  <link rel="stylesheet" href="${styleUri}" />
  <title>Mass Spectrum</title>
</head>
<body>
  <div id="spectrum-app" class="spectrum-status">Select a spectrum record.</div>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
}

function activate(context) {
  outputChannel = vscode.window.createOutputChannel("msentity Spectrum Viewer");
  context.subscriptions.push(outputChannel);

  const provider = new MSEntityViewerProvider(context);
  context.subscriptions.push(provider);
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(VIEW_TYPE, provider, {
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: false
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("msentitySpectrumViewer.open", async (uri) => {
      const target = uri || vscode.window.activeTextEditor?.document?.uri;
      if (!target) {
        vscode.window.showWarningMessage("Select an .msds, .msp, or .mgf file first.");
        return;
      }
      await vscode.commands.executeCommand("vscode.openWith", target, VIEW_TYPE);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("msentitySpectrumViewer.showLogs", () => outputChannel.show(true))
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
