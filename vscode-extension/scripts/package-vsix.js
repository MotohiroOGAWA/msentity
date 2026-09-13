"use strict";

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const { createVSIX } = require("@vscode/vsce");
const manifest = require("../package.json");
const { buildRuntime, stageTarget, TARGETS } = require("./build-runtime");

const extensionRoot = path.resolve(__dirname, "..");
const outputDirectory = path.join(extensionRoot, "dist");
const runtimeDir = path.join(extensionRoot, "runtime");

// Platform-specific packages: each embeds a private Python runtime with
// msentity preinstalled, so installing the extension needs no separate
// Python setup. Only one target's runtime is ever staged into runtime/ (the
// folder actually shipped) at a time — see build-runtime.js's module
// comment for why @vscode/vsce's `ignoreOtherTargetFolders` can't be relied
// on for this instead.
const RELEASE_TARGETS = Object.keys(TARGETS);

// @vscode/vsce's secret-scanning pass (out/secretLint.js) calls
// `process.exit(1)` directly on any file-read error instead of throwing —
// observed intermittently (ENOENT) when scanning the ~7000+ files of a
// freshly staged runtime tree, for reasons not fully understood (the files
// are present before and after; possibly filesystem-consistency lag under
// this much I/O). That exit() bypasses our try/finally and kills the whole
// Node process, which would corrupt a shared multi-target loop. Packaging
// each target in its own child process contains that failure to one target
// and lets us retry it without redoing the others or losing runtime/'s
// cleanup.
const PACKAGE_ATTEMPTS = 3;

async function packageTarget(target) {
  const suffix = target ? `-${target}` : "";
  const versionedPackage = path.join(outputDirectory, `${manifest.name}-${manifest.version}${suffix}.vsix`);
  await createVSIX({
    cwd: extensionRoot,
    packagePath: versionedPackage,
    target,
    allowPackageAllSecrets: true,
    allowPackageEnvFile: true,
  });
  return versionedPackage;
}

function packageTargetInChildProcess(target) {
  for (let attempt = 1; attempt <= PACKAGE_ATTEMPTS; attempt += 1) {
    try {
      execFileSync(process.execPath, [__filename, `--package-one=${target}`], { stdio: "inherit" });
      return;
    } catch (error) {
      if (attempt === PACKAGE_ATTEMPTS) throw error;
      console.warn(`[package-vsix] Packaging ${target} failed on attempt ${attempt}/${PACKAGE_ATTEMPTS}; retrying.`);
    }
  }
}

async function main() {
  fs.mkdirSync(outputDirectory, { recursive: true });

  const packageOneArg = process.argv.find((arg) => arg.startsWith("--package-one="));
  if (packageOneArg) {
    // Invoked as an isolated child process for exactly one target; see
    // packageTargetInChildProcess above.
    const target = packageOneArg.slice("--package-one=".length);
    const versionedPackage = await packageTarget(target);
    fs.copyFileSync(versionedPackage, path.join(outputDirectory, `${manifest.name}-${target}.vsix`));
    return;
  }

  if (process.argv.includes("--release")) {
    try {
      for (const target of RELEASE_TARGETS) {
        buildRuntime([target]);
        stageTarget(target);
        packageTargetInChildProcess(target);
      }
    } finally {
      // Leave no bundled runtime behind for plain `npm run package` afterward.
      fs.rmSync(runtimeDir, { recursive: true, force: true });
    }
  } else {
    await packageTarget(undefined);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
