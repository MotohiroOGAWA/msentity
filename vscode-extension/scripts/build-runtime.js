"use strict";

// Assembles a self-contained CPython runtime (interpreter + msentity + its
// dependencies) for each packaging target.
//
// @vscode/vsce's `ignoreOtherTargetFolders` package option looks like it
// would let every target's runtime live side-by-side and have vsce strip the
// others per package, but as of @vscode/vsce@3.9.2 that option is only wired
// into the CLI flag parsing, not into the actual file-collection logic (see
// out/package.js) — so it silently packages every folder regardless of its
// name. To keep each platform's VSIX containing only its own runtime, this
// module builds each target into a cache directory, and
// scripts/package-vsix.js copies exactly one target's build into the single
// `runtime/` folder that actually gets shipped, immediately before packaging
// that target.
//
// Everything under `runtime/` and `.runtime-cache/` is generated; both are
// git-ignored and .vscodeignore-exempted (the ignore file only needs to keep
// the cache out, since `runtime/` must ship inside the VSIX).

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

// Pinned astral-sh/python-build-standalone release. Bump deliberately and
// re-verify `sitePackages`/`executable` below still match if this changes.
const PBS_RELEASE = "20260901";
const PYTHON_VERSION = "3.11.16";
const PIP_PYTHON_VERSION = "311";
const PIP_IMPLEMENTATION = "cp";
const PIP_ABI = "cp311";

// msentity's runtime dependencies (see ../../pyproject.toml), pinned to exact
// versions so both platform bundles ship identical dependency versions.
// Pinned (rather than "latest") because pip resolves each target's cross-
// platform install independently: newer pyarrow/numpy/pandas/h5py releases
// dropped the manylinux2014 tag pip's --platform manylinux2014_x86_64 alias
// matches, so an unpinned install silently picked different versions per
// platform. These are the newest releases as of this writing that still
// publish wheels for both manylinux2014_x86_64 and win_amd64/cp311. Bump
// deliberately, verifying both platforms resolve the same version.
// Full transitive deps (e.g. pandas' python-dateutil) are resolved too, by
// omitting --no-deps here; msentity itself is installed separately with
// --no-deps since these are already satisfied.
const RUNTIME_DEPENDENCIES = [
  "numpy==2.2.6",
  "pandas==2.3.2",
  "h5py==3.14.0",
  "pyarrow==20.0.0",
  "tqdm==4.70.0",
];

const TARGETS = {
  "linux-x64": {
    asset: `cpython-${PYTHON_VERSION}+${PBS_RELEASE}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz`,
    sitePackages: ["python", "lib", "python3.11", "site-packages"],
    pipPlatform: "manylinux2014_x86_64",
    pruneDirs: [["python", "include"], ["python", "share"]],
  },
  "win32-x64": {
    asset: `cpython-${PYTHON_VERSION}+${PBS_RELEASE}-x86_64-pc-windows-msvc-install_only_stripped.tar.gz`,
    sitePackages: ["python", "Lib", "site-packages"],
    pipPlatform: "win_amd64",
    pruneDirs: [["python", "include"], ["python", "libs"], ["python", "share"]],
  },
};

const extensionRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(extensionRoot, "..");
const cacheDir = path.join(extensionRoot, ".runtime-cache");
// The folder actually included in a package (see extension.js's
// resolvePythonExecutable and scripts/package-vsix.js). Only ever holds one
// target's runtime at a time.
const runtimeDir = path.join(extensionRoot, "runtime");
// Per-target build cache, so repeated release builds don't redo the pip
// install work (pip's own cache still applies, but this also skips the
// tarball re-extraction and pruning).
const targetCacheRoot = (targetName) => path.join(cacheDir, "runtime", targetName);

function run(command, args, options = {}) {
  console.log(`+ ${command} ${args.join(" ")}`);
  execFileSync(command, args, { stdio: "inherit", ...options });
}

function downloadAsset(assetName) {
  const destination = path.join(cacheDir, assetName);
  if (fs.existsSync(destination)) return destination;
  fs.mkdirSync(cacheDir, { recursive: true });
  const url = `https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/${assetName}`;
  const partial = `${destination}.part`;
  run("curl", ["-sL", "--fail", "-o", partial, url]);
  fs.renameSync(partial, destination);
  return destination;
}

function removeRecursive(target) {
  fs.rmSync(target, { recursive: true, force: true });
}

function pruneTree(root, relativeDirs) {
  for (const relativeParts of relativeDirs) {
    removeRecursive(path.join(root, ...relativeParts));
  }
  // __pycache__/*.pyc are regenerated on demand and are not needed on disk.
  const stack = [root];
  while (stack.length) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const entryPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === "__pycache__") removeRecursive(entryPath);
        else stack.push(entryPath);
      }
    }
  }
}

function buildMsentityWheel(wheelDir) {
  fs.mkdirSync(wheelDir, { recursive: true });
  run("python3", ["-m", "pip", "wheel", repoRoot, "--no-deps", "-w", wheelDir]);
  const wheel = fs.readdirSync(wheelDir).find((name) => name.startsWith("msentity-") && name.endsWith(".whl"));
  if (!wheel) throw new Error(`Could not find a built msentity wheel in ${wheelDir}`);
  return path.join(wheelDir, wheel);
}

function buildTarget(targetName, msentityWheel) {
  const target = TARGETS[targetName];
  if (!target) throw new Error(`Unknown runtime target: ${targetName}`);

  const targetRoot = targetCacheRoot(targetName);
  removeRecursive(targetRoot);
  fs.mkdirSync(targetRoot, { recursive: true });

  const archive = downloadAsset(target.asset);
  run("tar", ["-xzf", archive, "-C", targetRoot]);

  const sitePackages = path.join(targetRoot, ...target.sitePackages);
  const crossInstallFlags = [
    "--platform", target.pipPlatform,
    "--python-version", PIP_PYTHON_VERSION,
    "--implementation", PIP_IMPLEMENTATION,
    "--abi", PIP_ABI,
    "--only-binary=:all:",
  ];
  run("python3", [
    "-m", "pip", "install",
    "--target", sitePackages,
    ...crossInstallFlags,
    ...RUNTIME_DEPENDENCIES,
  ]);
  run("python3", [
    "-m", "pip", "install",
    "--no-deps",
    "--target", sitePackages,
    ...crossInstallFlags,
    msentityWheel,
  ]);

  pruneTree(targetRoot, target.pruneDirs);
  console.log(`[build-runtime] ${targetName} ready at ${targetRoot}`);
}

function buildRuntime(targetNames = Object.keys(TARGETS)) {
  const wheelDir = path.join(cacheDir, "msentity-wheel");
  removeRecursive(wheelDir);
  const msentityWheel = buildMsentityWheel(wheelDir);
  for (const targetName of targetNames) buildTarget(targetName, msentityWheel);
}

// Copies one target's cached build into runtime/, the folder actually
// shipped in a package. Must be called (and the result packaged) one target
// at a time — see the module comment above for why.
function stageTarget(targetName) {
  const targetRoot = targetCacheRoot(targetName);
  if (!fs.existsSync(targetRoot)) throw new Error(`Runtime for ${targetName} was not built yet; call buildRuntime() first.`);
  removeRecursive(runtimeDir);
  fs.cpSync(targetRoot, runtimeDir, { recursive: true });
}

module.exports = { buildRuntime, stageTarget, TARGETS };

if (require.main === module) {
  const requested = process.argv.slice(2);
  buildRuntime(requested.length ? requested : undefined);
}
