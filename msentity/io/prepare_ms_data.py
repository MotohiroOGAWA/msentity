# from ..core import MSDataset
from .. import MSDataset
from ..processing.id import set_spec_id
from .msp import read_msp
from .mgf import read_mgf

def load_ms_dataset(input_file:str, *, file_type:str = None, spec_id_prefix: str = None) -> MSDataset:
    if file_type is None:
        if input_file.endswith('.msp'):
            file_type = 'msp'
        elif input_file.endswith('.mgf'):
            file_type = 'mgf'
        elif input_file.endswith('.msds') or input_file.endswith('.hdf5') or input_file.endswith('.h5'):
            file_type = 'msds'
        else:
            raise ValueError("Cannot determine file type from extension. Please specify 'file_type' parameter.")
        
    if file_type == 'msp':
        dataset = read_msp(input_file, spec_id_prefix=spec_id_prefix)
    elif file_type == 'mgf':
        dataset = read_mgf(input_file, spec_id_prefix=spec_id_prefix)
    elif file_type == 'msds':
        dataset = MSDataset.load(input_file)
    else:
        raise ValueError("Unsupported file type. Use 'msp', 'mgf' or 'msds'.")

    if 'SpecID' not in dataset.columns and spec_id_prefix is not None:
        set_spec_id(dataset=dataset, prefix=spec_id_prefix)

    
    return dataset