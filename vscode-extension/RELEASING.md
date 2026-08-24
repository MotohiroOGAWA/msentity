# Releasing the VS Code extension

This guide publishes a specific version of `msentity Spectrum Viewer` to
GitHub Releases. A release contains two identical VSIX packages with different
filenames:

- `msentity-spectrum-viewer-<version>.vsix` supports version-pinned downloads.
- `msentity-spectrum-viewer.vsix` supports the stable
  [`latest/download` URL](https://github.com/MotohiroOGAWA/msentity/releases/latest/download/msentity-spectrum-viewer.vsix).

## Prerequisites

Install Node.js, npm, Git, and the GitHub CLI (`gh`). Authenticate the GitHub
CLI before starting:

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

Install the exact dependencies recorded in `package-lock.json`, then build both
VSIX filenames:

```bash
npm ci
npm run package:release
```

For the selected version, this produces the following files under the ignored
`dist/` directory:

```text
dist/msentity-spectrum-viewer-<version>.vsix
dist/msentity-spectrum-viewer.vsix
```

Verify that both files exist and are identical:

```bash
ls -lh "dist/msentity-spectrum-viewer-${RELEASE_VERSION}.vsix" dist/msentity-spectrum-viewer.vsix
cmp "dist/msentity-spectrum-viewer-${RELEASE_VERSION}.vsix" dist/msentity-spectrum-viewer.vsix
```

Optionally install the versioned package locally for a smoke test:

```bash
code --install-extension "./dist/msentity-spectrum-viewer-${RELEASE_VERSION}.vsix" --force
```
or
```bash
code --install-extension "./dist/msentity-spectrum-viewer.vsix" --force
```

Open an MSDS, MSP, or MGF file and confirm that the viewer loads correctly.

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

Create the GitHub Release from that tag and upload both assets:

```bash
gh release create "$RELEASE_TAG" \
  "dist/msentity-spectrum-viewer-${RELEASE_VERSION}.vsix" \
  dist/msentity-spectrum-viewer.vsix \
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

Inspect the release and confirm that both assets are listed:

```bash
gh release view "$RELEASE_TAG" --repo MotohiroOGAWA/msentity
```

Verify the version-pinned download:

```bash
curl --fail --location --output /tmp/msentity-spectrum-viewer.vsix \
  "https://github.com/MotohiroOGAWA/msentity/releases/download/${RELEASE_TAG}/msentity-spectrum-viewer-${RELEASE_VERSION}.vsix"
```

Verify that the stable URL now resolves to the newly published release:

```bash
curl --fail --location --output /tmp/msentity-spectrum-viewer-latest.vsix \
  https://github.com/MotohiroOGAWA/msentity/releases/latest/download/msentity-spectrum-viewer.vsix
```

Do not publish the release as a prerelease if it should become the target of
the `releases/latest/download` URL.
