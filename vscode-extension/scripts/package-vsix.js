"use strict";

const fs = require("fs");
const path = require("path");
const { createVSIX } = require("@vscode/vsce");
const manifest = require("../package.json");
const { buildRuntime, TARGETS } = require("./build-runtime");

const extensionRoot = path.resolve(__dirname, "..");
const outputDirectory = path.join(extensionRoot, "dist");

// Platform-specific packages: each embeds a private Python runtime with
// msentity preinstalled, so installing the extension needs no separate
// Python setup. `ignoreOtherTargetFolders` makes vsce strip every
// `runtime/<other-target>/` folder from a given package, so one runtime
// build (see build-runtime.js) can back every target below.
const RELEASE_TARGETS = Object.keys(TARGETS);

async function packageTarget(target) {
  const suffix = target ? `-${target}` : "";
  const versionedPackage = path.join(outputDirectory, `${manifest.name}-${manifest.version}${suffix}.vsix`);
  await createVSIX({
    cwd: extensionRoot,
    packagePath: versionedPackage,
    target,
    ignoreOtherTargetFolders: Boolean(target),
  });
  return versionedPackage;
}

async function main() {
  fs.mkdirSync(outputDirectory, { recursive: true });

  if (process.argv.includes("--release")) {
    buildRuntime(RELEASE_TARGETS);
    for (const target of RELEASE_TARGETS) {
      const versionedPackage = await packageTarget(target);
      fs.copyFileSync(versionedPackage, path.join(outputDirectory, `${manifest.name}-${target}.vsix`));
    }
  } else {
    await packageTarget(undefined);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
