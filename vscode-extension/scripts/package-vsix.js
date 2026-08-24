"use strict";

const fs = require("fs");
const path = require("path");
const { createVSIX } = require("@vscode/vsce");
const manifest = require("../package.json");

const extensionRoot = path.resolve(__dirname, "..");
const outputDirectory = path.join(extensionRoot, "dist");
const versionedPackage = path.join(
  outputDirectory,
  `${manifest.name}-${manifest.version}.vsix`
);

async function main() {
  fs.mkdirSync(outputDirectory, { recursive: true });
  await createVSIX({ cwd: extensionRoot, packagePath: versionedPackage });

  if (process.argv.includes("--release")) {
    fs.copyFileSync(
      versionedPackage,
      path.join(outputDirectory, `${manifest.name}.vsix`)
    );
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
