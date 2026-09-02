#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_dir', default='preds')
    parser.add_argument('--out', default='challenges/kaggle-knee-mri/preds_oof.csv')
    args = parser.parse_args()
    files = list(Path(args.preds_dir).glob('*.npy'))
    rows = []
    for f in files:
        obj = np.load(f, allow_pickle=True).item()
        ids = obj['ids']
        probs = obj['probs']
        for id_, prob in zip(ids, probs):
            # prob may be array
            rows.append({'id': id_, 'prob': prob.tolist() if hasattr(prob, 'tolist') else float(prob)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print('Saved OOF to', args.out)

if __name__ == '__main__':
    main()
