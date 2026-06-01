from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

DATA_DIR = TESTS_DIR / "test_data"

SAMPLE_MSP_FILE = DATA_DIR / "sample.msp"
SAMPLE_MGF_FILE = DATA_DIR / "sample.mgf"
SAMPLE_HDF5_FILE = DATA_DIR / "sample.hdf5"