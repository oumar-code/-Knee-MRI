import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from challenges.kaggle_knee_mri.data.dataset import StudySliceDataset
from challenges.kaggle_knee_mri.models.baseline import SliceModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--files', type=str, required=True, help='CSV with study_path,group')
    p.add_argument('--n_slices', type=int, default=16)
    p.add_argument('--out', type=str, default='preds.npz')
    return p.parse_args()


def collate_fn(batch):
    xs = [b[0] for b in batch]
    xs = torch.stack(xs)
    B, S, H, W = xs.shape
    xs = xs.unsqueeze(2)
    return xs


def infer(args):
    df = np.genfromtxt(args.files, dtype=str, delimiter=',', names=True)
    entries = df['study_path'].tolist()
    dummy_labels = [[0]*12 for _ in entries]
    ds = StudySliceDataset(entries, dummy_labels, n_slices=args.n_slices)
    loader = DataLoader(ds, batch_size=8, collate_fn=lambda b: collate_fn(b), num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SliceModel(pretrained=False, n_outputs=12, in_channels=1)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            out = model(x)
            preds.append(torch.sigmoid(out).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    np.savez_compressed(args.out, preds=preds)


if __name__ == '__main__':
    args = parse_args()
    infer(args)
