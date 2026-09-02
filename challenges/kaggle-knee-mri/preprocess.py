"""Preprocess DICOM studies into per-study .npz volumes.

Usage example:
    python challenges/kaggle-knee-mri/preprocess.py \
        --dicom-root /path/to/raw_dicom_root \
        --out-dir data/preprocessed \
        --target-size 320 320 \
        --keywords sagittal,PD

What this script does:
- Walks the given dicom_root and groups DICOM files by StudyInstanceUID and SeriesInstanceUID.
- For each study, selects one series using a keyword-ranked heuristic (or the largest series if none match).
- Loads the selected series (pixel arrays), applies percentile clipping + z-score normalization, resizes slices to target size (if Pillow is available), and saves each study as a compressed .npz file containing an array shape (S, H, W).

Notes:
- The loader currently sorts files by InstanceNumber if available, else by filename. This should be robust but may be adjusted to use ImagePositionPatient for better ordering.
- The script attempts to be resilient: it will skip series that fail to read and continue.
"""

from pathlib import Path
import argparse
import pydicom
import numpy as np
from collections import defaultdict
import sys
import math

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def load_series_files(file_paths):
    """Load list of DICOM files into a numpy volume (S, H, W).
    Order by InstanceNumber when available, else filename.
    """
    slices = []
    meta_list = []
    for p in sorted(file_paths):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=False)
            arr = ds.pixel_array.astype('float32')
            instance = getattr(ds, 'InstanceNumber', None)
            slices.append((instance, arr, p))
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}", file=sys.stderr)
    if len(slices) == 0:
        raise RuntimeError('No readable DICOM slices in series')
    # sort by InstanceNumber if available
    if any(s[0] is not None for s in slices):
        slices = sorted(slices, key=lambda x: (x[0] if x[0] is not None else 0))
    else:
        # already sorted by filename
        pass
    arrs = [s[1] for s in slices]
    vol = np.stack(arrs, axis=0)
    return vol


def normalize_volume(vol, clip_percentiles=(0.5, 99.5)):
    p0, p1 = np.percentile(vol, clip_percentiles)
    vol = np.clip(vol, p0, p1)
    # z-score per-volume
    mean = vol.mean()
    std = vol.std()
    if std < 1e-6:
        std = 1.0
    vol = (vol - mean) / std
    return vol


def resize_volume(vol, target_h, target_w):
    if not PIL_AVAILABLE:
        # fallback: center-crop or pad to target
        S, H, W = vol.shape
        out = np.zeros((S, target_h, target_w), dtype=vol.dtype)
        for i in range(S):
            slice_i = vol[i]
            # simple center-crop or pad
            start_h = max(0, (H - target_h) // 2)
            start_w = max(0, (W - target_w) // 2)
            crop = slice_i[start_h:start_h+target_h, start_w:start_w+target_w]
            ch, cw = crop.shape
            out[i, :ch, :cw] = crop
        return out
    S, H, W = vol.shape
    out = np.zeros((S, target_h, target_w), dtype='float32')
    for i in range(S):
        arr = vol[i]
        # scale arr to 0-255 for PIL
        amin = float(arr.min())
        amax = float(arr.max())
        if (amax - amin) < 1e-6:
            img = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
        else:
            scaled = (arr - amin) / (amax - amin)
            scaled = (scaled * 255.0).astype('uint8')
            img = Image.fromarray(scaled)
        img = img.resize((target_w, target_h), resample=Image.BILINEAR)
        arr2 = np.asarray(img).astype('float32') / 255.0
        # restore approximate original scale (keep normalized behavior)
        out[i] = arr2
    return out


def find_dicom_series(dicom_root: Path):
    """Walk dicom_root and group DICOM files into studies and series.

    Returns: studies: dict[study_uid] -> dict[series_uid] -> {'desc':..., 'files':[Path,...]}
    """
    studies = defaultdict(lambda: defaultdict(lambda: {'desc': None, 'files': []}))
    for p in dicom_root.rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.dcm', ''):
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=False)
                study_uid = getattr(ds, 'StudyInstanceUID', None)
                series_uid = getattr(ds, 'SeriesInstanceUID', None)
                series_desc = getattr(ds, 'SeriesDescription', '')
                if study_uid is None or series_uid is None:
                    continue
                studies[study_uid][series_uid]['desc'] = series_desc
                studies[study_uid][series_uid]['files'].append(p)
            except Exception:
                # ignore files that are not DICOM or cannot be parsed
                continue
    return studies


def pick_series_for_study(series_dict, keywords=None):
    """Pick a series UID from series_dict using keywords ranking; fallback to largest series.

    keywords: list of lowercase substrings to prefer (order defines priority)
    """
    if keywords is None:
        keywords = []
    # score series by keyword matches
    scores = []
    for sid, info in series_dict.items():
        desc = (info.get('desc') or '').lower()
        score = 0
        for i, kw in enumerate(keywords):
            if kw in desc:
                score = len(keywords) - i  # earlier keywords higher score
                break
        scores.append((score, len(info['files']), sid))
    # choose max by (score, n_files)
    scores = sorted(scores, key=lambda x: (x[0], x[1]), reverse=True)
    if len(scores) == 0:
        return None
    return scores[0][2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dicom-root', type=str, required=True)
    p.add_argument('--out-dir', type=str, required=True)
    p.add_argument('--keywords', type=str, default='sagittal,pd', help='Comma-separated preferred series keywords (case-insensitive)')
    p.add_argument('--target-size', type=int, nargs=2, default=[320, 320], help='Target H W for slices')
    p.add_argument('--clip-percentiles', type=float, nargs=2, default=[0.5, 99.5])
    p.add_argument('--max-studies', type=int, default=0, help='Process at most N studies (0 = all)')
    p.add_argument('--force', action='store_true', help='Overwrite existing outputs')
    args = p.parse_args()

    dicom_root = Path(args.dicom_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keywords = [k.strip().lower() for k in args.keywords.split(',') if k.strip()]

    print('Scanning DICOM files (this may take a while)...')
    studies = find_dicom_series(dicom_root)
    print(f'Found {len(studies)} studies')

    processed = 0
    errors = 0
    for study_uid, series_map in studies.items():
        if args.max_studies and processed >= args.max_studies:
            break
        out_path = out_dir / f'{study_uid}.npz'
        if out_path.exists() and not args.force:
            print(f'Skipping {study_uid} (already exists)')
            continue
        sid = pick_series_for_study(series_map, keywords=keywords)
        if sid is None:
            print(f'No series found for study {study_uid}, skipping')
            continue
        files = series_map[sid]['files']
        try:
            vol = load_series_files(files)
            vol = normalize_volume(vol, clip_percentiles=tuple(args.clip_percentiles))
            if args.target_size is not None:
                vol = resize_volume(vol, args.target_size[0], args.target_size[1])
            # save as compressed npz
            np.savez_compressed(str(out_path), vol)
            processed += 1
            if processed % 50 == 0:
                print(f'Processed {processed} studies...')
        except Exception as e:
            print(f'Failed to process study {study_uid}: {e}', file=sys.stderr)
            errors += 1

    print(f'Done. Processed {processed} studies, errors: {errors}')
    if not PIL_AVAILABLE:
        print('Note: Pillow is not available, resize used a fallback method. Consider installing pillow for higher-quality resizing.')


if __name__ == '__main__':
    main()
