# msentity

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)  
![Python](https://img.shields.io/badge/Python-3.10+-blue)  
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://msentity.readthedocs.io)

---

**msentity** is a lightweight Python toolkit for reading, representing, and
manipulating mass-spectrometry datasets. It combines pandas-based spectrum
metadata with compact peak storage and provides MSP, MGF, and native MSDS I/O.

Use it from Python, the command line, an interactive dataset shell, or the
optional browser-based spectrum viewer.

---

## Documentation

Full documentation is available at:

https://msentity.readthedocs.io

---

## Basic installation

Install directly from GitHub:

```console
python -m pip install "msentity @ git+https://github.com/MotohiroOGAWA/msentity.git"
```

This installs the command-line tools without Gradio or the optional graphical
viewer.

## Installation with GUI

Install the GUI extra directly from GitHub:

```console
python -m pip install "msentity[gui] @ git+https://github.com/MotohiroOGAWA/msentity.git"
```

From an existing clone, use:

```console
python -m pip install ".[gui]"
```

The GUI extra installs Gradio and the custom spectrum-viewer component. The
regular installation remains smaller and includes every Python API, CLI
command, and the interactive shell.

## Command-line usage

Inspect or convert a dataset without writing Python:

```console
msentity info sample.msp
msentity head sample.msp --num-rows 5
msentity convert sample.msp sample.msds
```

Run `msentity --help` to see all commands, including directory merging and
MSDS metadata inspection.

## Interactive shell

Open an MSP, MGF, or MSDS file in a dataset-oriented prompt:

```console
msentity shell sample.msp
```

The shell is intended for quick inspection and preprocessing. Its commands
include `info`, `head`, `show`, `peaks`, `filter`, `sort`, `normalize`,
`specid`, `peakid`, `reset`, and `export`.

```text
msentity shell
Type 'help' to show commands. Type 'exit' to quit.

msentity> info
msentity> filter PrecursorMZ > 300
msentity> peaks 0 --top 10 --sort intensity
msentity> export filtered.msds
```

Use `help <command>` inside the shell for command-specific options. Filtering
and sorting operate on the current dataset view; `reset` restores the complete
view.

## GUI usage

Open an MSP, MGF, or MSDS dataset in the browser-based spectrum viewer:

```console
msentity gui sample.msp
msentity gui sample.mgf
msentity gui sample.msds
```

The viewer opens in a browser and shows a spectrum plot alongside its metadata.
Use the **Previous** and **Next** buttons or enter a zero-based spectrum index
to move through the dataset.

The server listens on `127.0.0.1:7860` by default. For example:

```console
# Container or remote host
msentity gui sample.msds --host 0.0.0.0 --port 7860 --no-browser

# Ask Gradio to create a temporary share URL
msentity gui sample.msds --share
```

Run `msentity gui --help` for all viewer options. A GUI screenshot can be added
later without changing the installation or usage sections above.
