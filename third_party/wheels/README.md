# Custom component wheels

This directory keeps the source artifact used to publish the optional
`gradio-msentityviewer` dependency. Normal installations download the immutable
GitHub Release asset; they do not install a wheel from this directory.

To publish a new component version:

1. Build `gradio_msentityviewer`.
2. Copy `dist/gradio_msentityviewer-0.1.0-py3-none-any.whl` here.
3. Inspect its `METADATA` and confirm the distribution name and version.
4. Create the GitHub Release tag `gradio-msentityviewer-v0.1.0`.
5. Upload the wheel as a Release asset.
6. Calculate its SHA256 with `sha256sum`.
7. Update the version, immutable Release URL, and `#sha256=` fragment in
   `pyproject.toml`.

When updating the component, update the wheel version, Release tag, and direct
URL together. Do not use a mutable `main` branch raw URL as a dependency.
