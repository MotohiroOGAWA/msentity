/* Run with Node and Playwright installed: node tests/test_similarity_ui.js */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vm = require("node:vm");
const http = require("node:http");
const { spawn, spawnSync } = require("node:child_process");
const chromium = process.env.WIZARD_ONLY ? null : require("playwright").chromium;
const root = path.resolve(__dirname, "../..");

async function wizardTests() {
  let choices = [], inputs = [], requests = [], messages = [];
  let selectedMode = "library_search";
  let browseReference = false;
  let writtenImage = null;
  const vscode = {
    Uri: {
      file: (fsPath) => ({ fsPath }),
      joinPath: (base, ...parts) => ({ fsPath: path.join(base.fsPath || String(base), ...parts) }),
    },
    workspace: { fs: { writeFile: async (target, bytes) => { writtenImage = { target, bytes }; } } },
    window: {
      showQuickPick: async (items, options) => {
        choices.push(options);
        if (options.title === "Calculate similarity") {
          return items.find((item) => item.mode === selectedMode);
        }
        if (options.title === "Library search: reference library" && browseReference) {
          return items.at(-1);
        }
        return items[0];
      },
      showInputBox: async (options) => {
        assert.ok(options.validateInput("NaN"));
        if (!options.title.includes("threshold")) assert.ok(options.validateInput("0"));
        assert.equal(options.validateInput(options.value), undefined);
        inputs.push(options); return options.value;
      },
      showOpenDialog: async () => [{ fsPath: "/tmp/reference.msp" }],
      showSaveDialog: async (options) => options.defaultUri,
      showInformationMessage: () => {},
    },
  };
  const context = { Buffer, require: (name) => name === "vscode" ? vscode : require(name), module: { exports: {} } };
  vm.runInNewContext(fs.readFileSync(path.join(root, "vscode-extension/extension.js"), "utf8") +
    "\nmodule.exports.Provider = MSEntityViewerProvider;", context);
  const provider = new context.module.exports.Provider({ extensionUri: "/extension" });
  const datasets = [0, 1, 2].map((i) => ({ id: `dataset${i}`, name: `dataset${i}`, columns: ["Name", "SpecID"] }));
  const run = (items, active = "dataset0") => provider.configureSimilarity(
    items, active, { fsPath: "/tmp/source.msds" },
    (m) => requests.push(m), (m) => messages.push(m), () => false
  );
  await run(datasets.slice(0, 2));
  assert.equal(choices.length, 5);
  assert.equal(inputs.length, 4);
  assert.equal(requests[0].mode, "library_search");
  assert.equal(requests[0].dataset1, "dataset0");
  assert.equal(requests[0].dataset2, "dataset1");
  assert.equal(requests[0].parameters.threshold, 0.8);
  assert.equal(requests[0].parameters.include_matched_data, true);

  choices = []; inputs = []; requests = [];
  selectedMode = "by_key";
  await run(datasets, "dataset1");
  assert.equal(requests[0].mode, "by_key");
  assert.equal(requests[0].dataset1, "dataset1");
  assert.equal(requests[0].dataset2, "dataset0");
  assert.equal(requests[0].parameters.key1, "SpecID");
  assert.equal(requests[0].parameters.key2, "SpecID");

  selectedMode = "library_search";
  browseReference = true;
  requests = [];
  await run(datasets.slice(0, 1));
  assert.equal(requests[0].reference_path, "/tmp/reference.msp");

  vscode.window.showQuickPick = async () => undefined;
  requests = []; messages = [];
  await run(datasets);
  assert.equal(requests.length, 0);
  assert.equal(messages[0].type, "similarity-cancelled");
  const savedPath = await provider.saveImage(
    { fsPath: "/tmp/data/result.mssim" },
    { filename: "similarity-box.png", bytes: [137, 80, 78, 71] }
  );
  assert.equal(savedPath, "/tmp/data/similarity-box.png");
  assert.deepEqual(Array.from(writtenImage.bytes), [137, 80, 78, 71]);
  vscode.window.showQuickPick = async (items, options) => { choices.push(options); return items[0]; };
  const savedPeaksPath = await provider.savePeaks(
    { fsPath: "/tmp/data/source.msds" },
    { basename: "upper-vs-lower", sourcePath: "/tmp/data/source.msds",
      contents: { tsv: "Upper m/z\tLower m/z\n100\t100.01\n", csv: "Upper m/z,Lower m/z\n100,100.01\n" } }
  );
  assert.equal(savedPeaksPath, "/tmp/data/upper-vs-lower.tsv");
  assert.equal(Buffer.from(writtenImage.bytes).toString("utf8"), "Upper m/z\tLower m/z\n100\t100.01\n");
  assert.equal(choices.at(-1).title, "Save spectrum peaks");
}

async function browserTests() {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "mssim-ui-"));
  const file = path.join(temporary, "sample.mssim");
  const python = process.env.MSENTITY_PYTHON || "python";
  const env = { ...process.env, PYTHONPATH: root };
  const fixture = spawnSync(python, ["-c", `
import pandas as pd
from msentity.similarity import SimilarityDataset
import sys
SimilarityDataset(pd.DataFrame({'SpecID': ['<script>bad</script>', 'B', 'C'],
    'index1': [0,1,2], 'index2': [2,0,1], 'cosine_similarity': [0.,0.89,1.]}),
    {'parameters': {'key1':'SpecID','key2':'SpecID'}, 'description':'Sample results'}).save(sys.argv[1])
`, file], { env, encoding: "utf8" });
  assert.equal(fixture.status, 0, fixture.stderr);
  const server = http.createServer((request, response) => {
    if (request.url === "/") {
      response.setHeader("Content-Type", "text/html");
      response.end(`<!doctype html><html><head><meta charset="UTF-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'self'; script-src 'nonce-test';">
        <link rel="stylesheet" href="/theme.css"><link rel="stylesheet" href="/similarity.css"></head><body>
        <div id="app" data-filename="sample.mssim"></div><script nonce="test">window.acquireVsCodeApi = () => ({ getState: () => null, setState: () => {}, postMessage: m => window.sendHostMessage(m) });</script>
        <script nonce="test" src="/similarity.js"></script></body></html>`);
    } else if (request.url === "/theme.css") {
      response.setHeader("Content-Type", "text/css");
      response.end(":root { --vscode-font-family: sans-serif; --vscode-font-size: 13px; --vscode-foreground: #ddd; --vscode-editor-background: #181818; --vscode-panel-border: #393939; --vscode-descriptionForeground: #aaa; --vscode-button-secondaryBackground: #333; --vscode-button-secondaryForeground: #eee; --vscode-input-background: #292929; --vscode-input-foreground: #eee; --vscode-errorForeground: #ff7777; }");
    } else if (["/similarity.js", "/similarity.css"].includes(request.url)) {
      response.setHeader("Content-Type", request.url.endsWith(".js") ? "text/javascript" : "text/css");
      response.end(fs.readFileSync(path.join(root, "vscode-extension/media", request.url.slice(1))));
    } else { response.statusCode = 404; response.end(); }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const browser = await chromium.launch({ headless: true,
    ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
  let child;
  const savedImages = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1200, height: 1000 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
    await page.exposeFunction("sendHostMessage", (message) => {
      if (message.type === "save-similarity-image") {
        savedImages.push(message);
        return;
      }
      child.stdin.write(JSON.stringify({ ...message, type: message.type === "page-request" ? "page" : message.type }) + "\n");
    });
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    child = spawn(python, [path.join(root, "vscode-extension/python/similarity_backend.py"), file, "--page-size", "2"], { env });
    let buffer = "";
    child.stdout.on("data", (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split("\n"); buffer = lines.pop();
      for (const line of lines) if (line.startsWith("MSENTITY_JSON:")) {
        const message = JSON.parse(line.slice("MSENTITY_JSON:".length));
        page.evaluate((data) => window.dispatchEvent(new MessageEvent("message", { data })), message).catch(() => {});
      }
    });
    await page.waitForSelector(".stats");
    assert.equal(await page.locator(".table-scroll tbody tr").count(), 2);
    assert.ok((await page.locator(".table-scroll").innerText()).includes("<script>bad</script>"));
    await page.locator("#next").click();
    await page.waitForFunction(() => document.querySelector("footer").textContent.includes("Page 2 / 2"));
    await page.locator("#filter").click();
    await page.locator("#add-filter").click();
    await page.locator("#apply").click();
    await page.waitForFunction(() => document.querySelector(".stats strong").textContent.startsWith("2"));
    assert.ok((await page.locator(".table-scroll").innerText()).includes("0.89"));
    await page.locator("#filter").click();
    await page.locator("#clear").click();
    await page.waitForFunction(() => document.querySelector(".stats strong").textContent.startsWith("3"));
    await page.locator("#bins").fill("2"); await page.locator("#bins").press("Tab");
    await page.waitForFunction(() => document.querySelectorAll("[data-bin]").length === 2);
    await page.locator(".distribution summary").click();
    const counts = await page.locator(".frequency-table tbody tr td:nth-child(2)").allTextContents();
    assert.deepEqual(counts, ["1", "2"]);
    assert.equal(await page.locator(".stats div").count(), 8);
    assert.ok((await page.locator(".stats").innerText()).includes("Q1"));
    assert.ok((await page.locator(".stats").innerText()).includes("Q3"));
    await page.locator("#count-scale").selectOption("log");
    assert.ok((await page.locator("#chart-hint").innerText()).includes("log₁₀"));
    await page.locator("#chart-type").selectOption("box");
    assert.ok((await page.locator(".distribution p").innerText()).includes("Q1"));
    await page.locator("#save-chart").click();
    await page.locator("#image-width").fill("640");
    await page.locator("#image-height").fill("480");
    await page.locator("#image-grid").uncheck();
    await page.locator("#image-box-color").fill("#ff0000");
    await page.locator("#confirm-chart").click();
    await page.waitForFunction(() => document.querySelector("#confirm-chart")?.textContent.includes("Preparing"));
    assert.equal(savedImages.length, 1);
    assert.equal(savedImages[0].filename, "similarity-box.png");
    assert.ok(savedImages[0].bytes.length > 1000);
    await page.evaluate(() => window.dispatchEvent(new MessageEvent("message", {
      data: { type: "image-save-complete", path: "/tmp/similarity-box.png" }
    })));
    await page.waitForFunction(() => !document.querySelector("#chart-export"));
    await page.locator("#chart-type").selectOption("histogram");
    await page.locator(".distribution summary").click();
    await page.locator('[data-bin="1"]').click();
    await page.waitForFunction(() => document.querySelector(".stats strong").textContent.startsWith("2"));
    assert.ok((await page.locator(".table-scroll").innerText()).includes("C")); // score 1 included
    await page.locator('[data-sort="cosine_similarity"]').click();
    await page.waitForSelector('[aria-sort="ascending"]');
    await page.waitForFunction(() => !document.querySelector("#reload").disabled);
    await page.locator('[data-sort="cosine_similarity"]').click();
    await page.waitForFunction(() => document.querySelector(".table-scroll tbody tr td").textContent === "C");
    await page.locator("#reload").click();
    await page.waitForFunction(() => !document.querySelector("#reload").disabled);
    await page.screenshot({ path: path.join(os.tmpdir(), "msentity-similarity-ui.png"), fullPage: true });
    await page.setViewportSize({ width: 500, height: 800 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    assert.deepEqual(errors, []);
  } finally {
    child?.kill(); await browser.close(); server.close(); fs.rmSync(temporary, { recursive: true, force: true });
  }
}

(async () => {
  await wizardTests();
  if (!process.env.WIZARD_ONLY) await browserTests();
  console.log(process.env.WIZARD_ONLY
    ? "Similarity wizard tests passed."
    : "Similarity wizard and browser interaction tests passed.");
})()
  .catch((error) => { console.error(error); process.exitCode = 1; });
