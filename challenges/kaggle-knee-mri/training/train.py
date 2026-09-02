import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import subprocess
import sys
import glob

from challenges.kaggle_knee_mri.data.dataset import StudySliceDataset
from challenges.kaggle_knee_mri.models.baseline import SliceModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default=None, help='CSV with columns: study_path, labels (as 12 semicolon-separated values), group')
    p.add_argument('--dicom-root', type=str, default=None, help='If provided, will run preprocessing to produce per-study .npz files')
    p.add_argument('--preprocessed-dir', type=str, default='data/preprocessed', help='Directory where .npz preprocessed volumes are stored or will be written')
    p.add_argument('--labels-csv', type=str, default=None, help='Optional CSV mapping StudyInstanceUID to 12 labels and group info (columns: study_uid, ACL, MCL, ..., group(optional))')
    p.add_argument('--keywords', type=str, default='sagittal,pd', help='Preferred series keywords passed to preprocess step')
    p.add_argument('--target-size', type=int, nargs=2, default=[320, 320])
    p.add_argument('--n_slices', type=int, default=16)
    p.add_argument('--backbone', type=str, default='tf_efficientnet_b3')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--outdir', type=str, default='checkpoints')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--auto-preprocess', action='store_true', help='If set and dicom-root provided, run preprocessing for missing studies')
    return p.parse_args()


def collate_fn(batch):
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    # xs are (C, H, W) per sample but here C == n_slices; we want (B, S, 1, H, W)
    xs = torch.stack(xs)  # (B, S, H, W)
    B, S, H, W = xs.shape
    xs = xs.unsqueeze(2)  # (B, S, 1, H, W)
    ys = torch.stack(ys)
    return xs, ys


def run_preprocess(dicom_root: str, out_dir: str, keywords: str, target_size: list):
    """Call the preprocess.py script to generate preprocessed .npz files.
    This runs the existing CLI script in the repository using the same Python interpreter.
    """
    script_path = os.path.join(os.path.dirname(__file__), '..', 'preprocess.py')
    script_path = os.path.abspath(script_path)
    cmd = [sys.executable, script_path, '--dicom-root', dicom_root, '--out-dir', out_dir,
           '--keywords', keywords, '--target-size', str(target_size[0]), str(target_size[1])]
    print('Running preprocess:', ' '.join(cmd))
    subprocess.check_call(cmd)


def build_train_csv_from_preprocessed(preprocessed_dir: str, labels_csv: str = None, out_csv: str = 'train_entries.csv'):
    """Create a CSV with columns: study_path, labels (semicolon-separated 12), group

    If labels_csv is provided, it should contain a column 'study_uid' and the 12 label columns in the same order as the README.
    If group column is missing, group defaults to study_uid.
    If labels_csv is not provided, labels default to zeros (placeholder) and group=study_uid.
    """
    pre_dir = Path(preprocessed_dir)
    files = sorted(pre_dir.glob('*.npz'))
    rows = []
    labels_df = None
    if labels_csv is not None and Path(labels_csv).exists():
        labels_df = pd.read_csv(labels_csv)
        # expect study_uid column
        if 'study_uid' not in labels_df.columns:
            # try common alternatives
            if 'StudyInstanceUID' in labels_df.columns:
                labels_df = labels_df.rename(columns={'StudyInstanceUID': 'study_uid'})
            else:
                raise ValueError('labels_csv must contain study_uid or StudyInstanceUID column')
        labels_df = labels_df.set_index('study_uid')

    for f in files:
        study_uid = f.stem
        study_path = str(f)
        if labels_df is not None and study_uid in labels_df.index:
            row = labels_df.loc[study_uid]
            # attempt to collect 12 label columns (non-group)
            # if labels are already in single column 'labels' with semicolon, use that
            if 'labels' in row.index:
                lab_str = str(row['labels'])
            else:
                # pick numeric columns (not group)
                numeric_cols = [c for c in labels_df.columns if labels_df[c].dtype.kind in 'fi' or labels_df[c].dtype == object]
                # remove potential 'group' column if present
                if 'group' in numeric_cols:
                    numeric_cols.remove('group')
                vals = []
                # If exactly 12 columns are present, use first 12
                for c in numeric_cols[:12]:
                    vals.append(str(row[c]))
                lab_str = ';'.join(vals)
            group = row['group'] if 'group' in row.index else study_uid
        else:
            lab_str = ';'.join(['0'] * 12)
            group = study_uid
        rows.append({'study_path': study_path, 'labels': lab_str, 'group': group})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f'Wrote training CSV with {len(out_df)} rows to {out_csv}')
    return out_csv


def train_fold(args):
    # If auto-preprocess requested, run preprocessing for missing studies
    if args.auto_preprocess and args.dicom_root:
        os.makedirs(args.preprocessed_dir, exist_ok=True)
        try:
            run_preprocess(args.dicom_root, args.preprocessed_dir, args.keywords, args.target_size)
        except subprocess.CalledProcessError as e:
            print('Preprocessing step failed:', e)
            raise

    # If csv not provided, build from preprocessed dir (using optional labels_csv)
    if args.csv is None:
        os.makedirs(args.preprocessed_dir, exist_ok=True)
        generated_csv = os.path.join(args.preprocessed_dir, 'train_entries.csv')
        args.csv = build_train_csv_from_preprocessed(args.preprocessed_dir, args.labels_csv, out_csv=generated_csv)

    df = pd.read_csv(args.csv)
    # expect df columns: study_path, labels (as string '0;1;0;...'), group
    entries = df['study_path'].tolist()
    labels = np.stack(df['labels'].apply(lambda s: np.array([float(x) for x in s.split(';')])).values)
    groups = df['group'].values

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(entries, labels, groups))
    train_idx, val_idx = splits[args.fold]

    train_entries = [entries[i] for i in train_idx]
    val_entries = [entries[i] for i in val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    train_ds = StudySliceDataset(train_entries, train_labels, n_slices=args.n_slices)
    val_ds = StudySliceDataset(val_entries, val_labels, n_slices=args.n_slices)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SliceModel(backbone_name=args.backbone, pretrained=True, n_outputs=12, in_channels=1)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f'Train epoch {epoch}'):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
        print(f'Epoch {epoch} train loss: {running_loss / len(train_loader.dataset):.4f}')

        # quick validation (forward only)
        model.eval()
        preds = []
        gts = []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc='Val'):
                x = x.to(device)
                out = model(x)
                preds.append(torch.sigmoid(out).cpu().numpy())
                gts.append(y.numpy())
        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)
        # save OOF preds for stacking
        np.savez_compressed(f"{args.outdir}/oof_fold{args.fold}_epoch{epoch}.npz", preds=preds, gts=gts, val_idx=val_idx)
        torch.save(model.state_dict(), f"{args.outdir}/model_fold{args.fold}_epoch{epoch}.pth")


if __name__ == '__main__':
    args = parse_args()
    train_fold(args)
