import pydicom
import numpy as np
from pathlib import Path


def load_series(series_folder: Path):
    """Load a DICOM series folder and return a numpy volume (S, H, W).

    This is a simple loader that reads all .dcm files in a folder sorted by filename.
    It does not attempt to parse ImagePositionPatient for ordering — for many competitions
    filenames are ordered, but you should replace sorting with proper DICOM ordering
    if available.
    """
    files = sorted(list(series_folder.glob('*.dcm')))
    if len(files) == 0:
        raise FileNotFoundError(f"No DICOM files found in {series_folder}")
    slices = []
    for p in files:
        ds = pydicom.dcmread(str(p))
        arr = ds.pixel_array.astype('float32')
        slices.append(arr)
    volume = np.stack(slices, axis=0)
    return volume


def normalize_volume(vol: np.ndarray, clip_percentiles=(0.5, 99.5)):
    p0, p1 = np.percentile(vol, clip_percentiles)
    vol = np.clip(vol, p0, p1)
    vol = (vol - vol.mean()) / (vol.std() + 1e-8)
    return vol
