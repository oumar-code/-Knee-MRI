#!/usr/bin/env python3
"""Generate a simple CSV mapping study id -> npz path and (optionally) labels.
If labels.csv is available (columns: id,label_0..label_11), it will be merged.
"""
import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_root', default='data/preprocessed')
    parser.add_argument('--labels', default='')
    parser.add_argument('--out', default='challenges/kaggle-knee-mri/data/dataset.csv')
    args = parser.parse_args()
    p = Path(args.npz_root)
    files = list(p.glob('*.npz'))
    rows = []
    for f in files:
        rows.append({'id': f.stem, 'path': str(f)})
    df = pd.DataFrame(rows)
    if args.labels and Path(args.labels).exists():
        lab = pd.read_csv(args.labels)
        df = df.merge(lab, how='left', on='id')
    df.to_csv(args.out, index=False)
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
