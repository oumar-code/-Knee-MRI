#!/usr/bin/env python3
import argparse
import subprocess
import os
from pathlib import Path


def run_fold(fold, csv, extra_args=''):
    cmd = f"python3 challenges/kaggle-knee-mri/training/train.py --csv {csv} --fold {fold} --log_dir runs --checkpoint_dir models --batch_size 8 --n_slices 16 --max_epochs 50 --gpus 1 {extra_args}"
    print('Running:', cmd)
    subprocess.check_call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='challenges/kaggle-knee-mri/data/dataset.csv')
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    os.makedirs('preds', exist_ok=True)
    for f in range(args.folds):
        run_fold(f, args.csv)

if __name__ == '__main__':
    main()
