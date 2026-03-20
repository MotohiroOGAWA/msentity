from ..core import MSDataset
from ..processing.id import set_spec_id
from . import read_msp, read_mgf

def load_ms_data(input_file:str, *, file_type:str = None, spec_id_prefix: str = None) -> MSDataset:
    if file_type is None:
        if input_file.endswith('.msp'):
            file_type = 'msp'
        elif input_file.endswith('.mgf'):
            file_type = 'mgf'
        elif input_file.endswith('hdf5') or input_file.endswith('.h5'):
            file_type = 'hdf5'
        else:
            raise ValueError("Cannot determine file type from extension. Please specify 'file_type' parameter.")
        
    if file_type == 'msp':
        dataset = read_msp(input_file, spec_id_prefix=spec_id_prefix)
    elif file_type == 'mgf':
        dataset = read_mgf(input_file, spec_id_prefix=spec_id_prefix)
    elif file_type == 'hdf5':
        dataset = MSDataset.from_hdf5(input_file)
        if 'SpecID' not in dataset.columns:
            set_spec_id(dataset=dataset, prefix=spec_id_prefix)

    else:
        raise ValueError("Unsupported file type. Use 'msp', 'mgf' or 'hdf5'.")
    
    return dataset