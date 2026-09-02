import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class StudySliceDataset(Dataset):
    """Dataset that loads preprocessed numpy volumes saved as .npz or .npy

    Expects a list of entries where each entry points to a preprocessed file
    containing a volume with shape (S, H, W). The labels are a list/array
    of shape (12,) per study.
    """

    def __init__(self, entries, labels, n_slices=16, transform=None):
        self.entries = entries
        self.labels = labels
        self.n_slices = n_slices
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def _load_volume(self, path):
        path = Path(path)
        if path.suffix == '.npz':
            data = np.load(path)
            vol = data[list(data.files)[0]]
        else:
            vol = np.load(path)
        return vol.astype('float32')

    def __getitem__(self, idx):
        vol = self._load_volume(self.entries[idx])  # (S, H, W)
        L = vol.shape[0]
        if L >= self.n_slices:
            indices = np.linspace(0, L - 1, self.n_slices, dtype=int)
        else:
            indices = np.concatenate([np.arange(L), np.repeat(L - 1, self.n_slices - L)])
        slices = vol[indices]
        # normalize per-slice (optionally per-volume in preprocessing)
        # stack slices as channels: (C, H, W)
        x = np.stack([s for s in slices], axis=0)
        # convert to (C, H, W) floats
        x = torch.from_numpy(x).float()
        # add channel dim expected by 2D backbones: treat each slice as single-channel
        # many timm models expect 3-channel input; we'll repeat channels in model transform if needed
        y = torch.tensor(self.labels[idx]).float()
        if self.transform is not None:
            x = self.transform(x)
        return x, y
