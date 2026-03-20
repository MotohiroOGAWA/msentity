try:
    from splash import *
except ImportError:
    print("splash is not installed.")

from ..core import *

def compute_splash(peaks: SpectrumPeaks) -> str:
    _peaks = [(float(peak.mz), float(peak.intensity)) for peak in peaks]
    spectrum = Spectrum(_peaks, SpectrumType.MS)
    return Splash().splash(spectrum)

def compute_splash_batch(peak_series: PeakSeries) -> list[str]:
    splash_ids = []
    for peaks in peak_series:
        splash_id = compute_splash(peaks)
        splash_ids.append(splash_id)
    return splash_ids

def add_splash_to_dataset(dataset: MSDataset, splash_column: str = "SPLASH", in_place: bool = True) -> MSDataset:
    if in_place:
        ds = dataset
    else:
        ds = dataset.copy()
    
    splash_ids = compute_splash_batch(ds.peaks)
    ds[splash_column] = splash_ids
    return ds