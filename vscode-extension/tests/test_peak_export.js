const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "../media/spectrum.js"), "utf8");
const start = source.indexOf("  const peaksAsDelimited =");
const end = source.indexOf("  const matchSpectrumPeaks", start);
assert.ok(start >= 0 && end > start);
const context = vm.createContext({ state: { sortKey: "intensity", sortAscending: false } });
vm.runInContext(source.slice(start, end) + "\nthis.exportPeaks = peaksAsDelimited;", context);

for (const [format, separator] of [["csv", ","], ["tsv", "\t"]]) {
  test(`peak ${format} retains zero/negative peaks and follows table sorting`, () => {
    const actual = context.exportPeaks([100, 200, 300, NaN], [0, -2, 10, 5], format);
    assert.equal(actual, [
      ["m/z", "Intensity"], [300, 10], [100, 0], [200, -2],
    ].map(row => row.join(separator)).join("\n") + "\n");
  });
}
