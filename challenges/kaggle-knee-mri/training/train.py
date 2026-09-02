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

from challenges.kaggle_knee_mri.data.dataset import StudySliceDataset
from challenges.kaggle_knee_mri.models.baseline import SliceModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, required=True, help='CSV with columns: study_path, labels (as 12 comma-separated values), group (StudyInstanceUID)')
    p.add_argument('--n_slices', type=int, default=16)
    p.add_argument('--backbone', type=str, default='tf_efficientnet_b3')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--outdir', type=str, default='checkpoints')
    p.add_argument('--fold', type=int, default=0)
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


def train_fold(args):
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
