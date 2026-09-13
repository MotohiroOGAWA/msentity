# Releasing the VS Code extension

This guide publishes a specific version of `msentity Spectrum Viewer` to
GitHub Releases. A release contains four VSIX packages: one versioned and one
fixed-name file for each of two platforms (`linux-x64` and `win32-x64`). Each
platform's package embeds a private Python runtime with `msentity`
preinstalled, so installing it requires no separate Python setup.

- `msentity-spectrum-viewer-<version>-<platform>.vsix` supports version-pinned
  downloads.
- `msentity-spectrum-viewer-<platform>.vsix` supports the stable
  `latest/download` URLs, e.g.
  [`.../latest/download/msentity-spectrum-viewer-linux-x64.vsix`](https://github.com/MotohiroOGAWA/msentity/releases/latest/download/msentity-spectrum-viewer-linux-x64.vsix).

Only `linux-x64` and `win32-x64` (x64) are built; ARM64 is not currently
supported.

## Prerequisites

Install Node.js, npm, Git, and the GitHub CLI (`gh`). Building the release
assets also requires a Python 3.9+ interpreter with `pip` and network access
to PyPI and GitHub on the build machine — this can be any single OS (e.g.
Linux), since `scripts/build-runtime.js` cross-installs the Windows wheels
too using pip's `--platform`/`--python-version`/`--abi` flags; no actual
Windows machine is needed to build the `win32-x64` package. Authenticate the
GitHub CLI before starting:

```bash
gh auth login
gh auth status
```

Run the following commands from the `vscode-extension` directory in a clean
checkout of the `main` branch:

```bash
cd vscode-extension
git status --short
```

## 1. Set the release version

Choose a version without a leading `v`. For example, to release version
`0.1.1`:

```bash
RELEASE_VERSION=0.1.1
npm version "$RELEASE_VERSION" --no-git-tag-version
```

This overwrites the `version` field in both `package.json` and
`package-lock.json`. The `--no-git-tag-version` option prevents npm from
creating a commit or Git tag; those are created explicitly later.

This command is the only place where the release version is entered. Do not
edit the repeated version fields in `package-lock.json` or put the current
version in the documentation manually. npm synchronizes the lock file,
`@vscode/vsce` reads the output version from `package.json`, and the commands
below reuse `RELEASE_VERSION` for filenames, the Git tag, and the release
title.

Confirm that both files contain the requested version:

```bash
node -p "require('./package.json').version"
node -p "require('./package-lock.json').version"
git diff -- package.json package-lock.json
```

## 2. Install dependencies and build the release assets

Install the exact dependencies recorded in `package-lock.json`, then build the
runtime bundles and both platforms' VSIX files:

```bash
npm ci
npm run package:release
```

This first runs `scripts/build-runtime.js`, which downloads a pinned CPython
3.11 build for each platform from `astral-sh/python-build-standalone` and
installs `msentity`'s pinned dependency versions into a git-ignored build
cache (`.runtime-cache/runtime/<platform>/`; this step needs network access
and takes a few minutes). It then packages each platform one at a time,
copying only that platform's cached runtime into the git-ignored `runtime/`
folder immediately before packaging so a given platform's VSIX contains only
its own runtime (`@vscode/vsce`'s `ignoreOtherTargetFolders` package option
looks built for this, but as of `@vscode/vsce@3.9.2` it is unimplemented —
see the comment at the top of `scripts/build-runtime.js`).

For the selected version, this produces the following files under the ignored
`dist/` directory:

```text
dist/msentity-spectrum-viewer-<version>-linux-x64.vsix
dist/msentity-spectrum-viewer-linux-x64.vsix
dist/msentity-spectrum-viewer-<version>-win32-x64.vsix
dist/msentity-spectrum-viewer-win32-x64.vsix
```

Verify that the versioned and fixed-name file are identical for each platform:

```bash
for PLATFORM in linux-x64 win32-x64; do
  ls -lh "dist/msentity-spectrum-viewer-${RELEASE_VERSION}-${PLATFORM}.vsix" "dist/msentity-spectrum-viewer-${PLATFORM}.vsix"
  cmp "dist/msentity-spectrum-viewer-${RELEASE_VERSION}-${PLATFORM}.vsix" "dist/msentity-spectrum-viewer-${PLATFORM}.vsix"
done
```

Optionally install the package matching this machine's OS locally for a smoke
test — on Linux:

```bash
code --install-extension "./dist/msentity-spectrum-viewer-${RELEASE_VERSION}-linux-x64.vsix" --force
```

or, after copying the `win32-x64` file to a Windows machine:

```console
code --install-extension .\msentity-spectrum-viewer-<version>-win32-x64.vsix --force
```

With `msentitySpectrumViewer.pythonPath` left unset and no system Python
installed, open an MSDS, MSP, MGF, TSV, and CSV file and confirm that the
viewer loads correctly using the bundled runtime (check "MS Entity: Show
Spectrum Viewer Logs" for the `[python]` line). Confirm that dataset export
and peak-table copy/save work for both TSV and CSV.

## 3. Commit the version update

Commit the manifest changes and push them to `main`:

```bash
git add package.json package-lock.json
git commit -m "Release msentity Spectrum Viewer v${RELEASE_VERSION}"
git push origin main
```

The generated VSIX files are release artifacts and should not be committed.

## 4. Create the GitHub Release

The repository uses tags in the form `msentity-v<version>`. Create and push an
annotated tag for the commit just pushed:

```bash
RELEASE_TAG="msentity-v${RELEASE_VERSION}"
git tag -a "$RELEASE_TAG" -m "msentity v${RELEASE_VERSION}"
git push origin "$RELEASE_TAG"
```

Create the GitHub Release from that tag and upload all four assets:

```bash
gh release create "$RELEASE_TAG" \
  "dist/msentity-spectrum-viewer-${RELEASE_VERSION}-linux-x64.vsix" \
  dist/msentity-spectrum-viewer-linux-x64.vsix \
  "dist/msentity-spectrum-viewer-${RELEASE_VERSION}-win32-x64.vsix" \
  dist/msentity-spectrum-viewer-win32-x64.vsix \
  --repo MotohiroOGAWA/msentity \
  --target main \
  --title "msentity v${RELEASE_VERSION}" \
  --generate-notes \
  --verify-tag
```

`--verify-tag` prevents an accidental release when the expected tag does not
exist. Alternatively, omit `--verify-tag` and let `gh release create` create
the tag at `--target main`.

## 5. Verify the published release

Inspect the release and confirm that all four assets are listed:

```bash
gh release view "$RELEASE_TAG" --repo MotohiroOGAWA/msentity
```

Verify the version-pinned downloads:

```bash
for PLATFORM in linux-x64 win32-x64; do
  curl --fail --location --output "/tmp/msentity-spectrum-viewer-${PLATFORM}.vsix" \
    "https://github.com/MotohiroOGAWA/msentity/releases/download/${RELEASE_TAG}/msentity-spectrum-viewer-${RELEASE_VERSION}-${PLATFORM}.vsix"
done
```

Verify that both stable URLs now resolve to the newly published release:

```bash
for PLATFORM in linux-x64 win32-x64; do
  curl --fail --location --output "/tmp/msentity-spectrum-viewer-${PLATFORM}-latest.vsix" \
    "https://github.com/MotohiroOGAWA/msentity/releases/latest/download/msentity-spectrum-viewer-${PLATFORM}.vsix"
done
```

Do not publish the release as a prerelease if it should become the target of
the `releases/latest/download` URLs.
