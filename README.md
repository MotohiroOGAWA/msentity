# msentity

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)  
![Python](https://img.shields.io/badge/Python-3.10+-blue)  
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://msentity.readthedocs.io)

---

**msentity** is a lightweight Python toolkit for representing and manipulating  
chemical entities in mass spectrometry workflows.

**msentity** aims to offer a clean and consistent API for downstream analysis and modeling.

---

## Documentation

Full documentation is available at:

👉 https://msentity.readthedocs.io

---

## Basic installation

Install directly from GitHub:

```bash
pip install git+https://github.com/MotohiroOGAWA/msentity.git
```

This installs the command-line tools without Gradio or the optional graphical
viewer.

## Installation with GUI

Install the GUI extra directly from GitHub:

```bash
pip install "msentity[gui] @ git+https://github.com/MotohiroOGAWA/msentity.git"
```

From an existing clone, use:

```bash
pip install ".[gui]"
```

## GUI usage

Open an MSP, MGF, or MSDS dataset in the browser-based spectrum viewer:

```bash
msentity gui sample.msp
msentity gui sample.mgf
msentity gui sample.msds
```

The server listens on `127.0.0.1:7860` and opens a browser by default. Use
`--host 0.0.0.0` for a container, `--port` to choose another port,
`--no-browser` to suppress browser launch, or `--share` to request a Gradio
share URL.
