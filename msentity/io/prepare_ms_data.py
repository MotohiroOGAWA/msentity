from typing import Union
from pathlib import Path

from .. import MSDataset
from ..processing.id import set_spec_id
from .msp import read_msp
from .mgf import read_mgf

def load_ms_dataset(
    input_file: Union[str, Path],
    *,
    file_type: str | None = None,
    spec_id_prefix: str | None = None,
) -> MSDataset:
    input_path = Path(input_file)

    if file_type is None:
        suffix = input_path.suffix.lower()

        if suffix == ".msp":
            file_type = "msp"
        elif suffix == ".mgf":
            file_type = "mgf"
        elif suffix in {".msds", ".hdf5", ".h5"}:
            file_type = "msds"
        else:
            raise ValueError(
                "Cannot determine file type from extension. "
                "Please specify 'file_type' parameter."
            )

    file_type = file_type.lower()

    if file_type == "msp":
        dataset = read_msp(input_path, spec_id_prefix=spec_id_prefix)
    elif file_type == "mgf":
        dataset = read_mgf(input_path, spec_id_prefix=spec_id_prefix)
    elif file_type == "msds":
        dataset = MSDataset.load(input_path)
    else:
        raise ValueError("Unsupported file type. Use 'msp', 'mgf' or 'msds'.")

    if "SpecID" not in dataset.columns and spec_id_prefix is not None:
        set_spec_id(dataset=dataset, prefix=spec_id_prefix)

    return dataset