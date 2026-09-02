import os
import math
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class StudySliceDataset(Dataset):
    """Dataset that loads preprocessed .npz (saved by preprocess.py).
    Each item returns:
      - image tensor: (S, 1, H, W) where S = n_slices sampled
      - label tensor: (C,) float32 (if labels provided)
      - id: study id (str)
    """
    def __init__(self, rows, n_slices=16, transforms=None):
        # rows: list of dicts with keys: id, path, labels (optional list/array)
        self.rows = rows
        self.n_slices = n_slices
        self.transforms = transforms

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        npz = np.load(r['path'], allow_pickle=True)
        if 'images' in npz:
            vol = npz['images']
        else:
            # fallback to first array
            vol = npz[next(iter(npz.files))]
        # vol shape: (Z, H, W)
        Z = vol.shape[0]
        if Z >= self.n_slices:
            indices = np.linspace(0, Z - 1, self.n_slices, dtype=int)
        else:
            # repeat last slice
            indices = np.concatenate([np.arange(Z), np.repeat(Z - 1, self.n_slices - Z)])
        slices = vol[indices]
        # normalize already done in preprocess; convert to tensor
        # add channel dim
        x = np.expand_dims(slices, axis=1)  # (S,1,H,W)
        x = torch.from_numpy(x).float()
        y = None
        if 'labels' in r and r['labels'] is not None:
            y = torch.tensor(r['labels'], dtype=torch.float32)
        return {'image': x, 'label': y, 'id': r['id']}


def collate_fn(batch):
    images = torch.stack([b['image'] for b in batch], dim=0)  # (B,S,1,H,W)
    labels = None
    ids = [b['id'] for b in batch]
    if batch[0]['label'] is not None:
        labels = torch.stack([b['label'] for b in batch], dim=0)
    return {'image': images, 'label': labels, 'id': ids}


class KneeDataModule:
    def __init__(self, csv_rows: List[dict], batch_size=8, num_workers=4, n_slices=16):
        self.csv_rows = csv_rows
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_slices = n_slices

    def setup(self, stage: Optional[str] = None):
        # Expect csv_rows to contain 'fold' key for splitting
        train_rows = [r for r in self.csv_rows if r.get('split') == 'train']
        val_rows = [r for r in self.csv_rows if r.get('split') == 'val']
        self.train_ds = StudySliceDataset(train_rows, n_slices=self.n_slices)
        self.val_ds = StudySliceDataset(val_rows, n_slices=self.n_slices)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)
