#!/usr/bin/env python3
"""Training script for the multimodal mid-fusion MRI model.

This script trains the mid-fusion model on a single fold of the Knee MRI dataset,
handling missing labels and reports gracefully. It saves the best checkpoint and
outputs OOF predictions for ensembling.

Usage:
    python train_multimodal.py \\
        --csv data/multimodal_train.csv \\
        --fold 0 \\
        --batch_size 8 \\
        --max_epochs 30 \\
        --output_dir models/multimodal
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from training.model import AttentionPool, compute_macro_auc
import timm


class MidFusionImageTextModel(pl.LightningModule):
    """Mid-fusion model: image encoder + text encoder + fusion head."""

    def __init__(
        self,
        n_outputs: int = 12,
        image_backbone: str = "tf_efficientnet_b3",
        embed_dim: int = 256,
        vocab_size: int = 5000,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Image encoder: backbone + attention pool + projection
        self.backbone = timm.create_model(
            image_backbone, pretrained=True, num_classes=0, global_pool=""
        )
        self.image_feat_dim = self.backbone.num_features
        self.image_pool = AttentionPool(self.image_feat_dim)
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_feat_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Text encoder: embedding + BiLSTM + projection
        self.text_embed = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.text_lstm = nn.LSTM(
            128, 256, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2
        )
        self.text_proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_outputs),
        )

        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        # Validation metrics
        self.val_targets = []
        self.val_preds = []

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            images: (B, S, 1, H, W) - MRI volume slices
            texts: (B, seq_len) - tokenized reports
            
        Returns:
            logits: (B, n_outputs)
        """
        # Image encoding
        B, S, C, H, W = images.shape
        x_img = images.view(B * S, C, H, W)
        if C == 1:
            x_img = x_img.repeat(1, 3, 1, 1)
        feats = self.backbone.forward_features(x_img)
        if feats.dim() == 4:
            feats = torch.mean(feats, dim=[2, 3])
        feats = feats.view(B, S, -1)
        img_pooled = self.image_pool(feats)
        img_emb = self.image_proj(img_pooled)

        # Text encoding
        txt_emb_in = self.text_embed(texts)
        _, (h_n, _) = self.text_lstm(txt_emb_in)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        txt_emb = self.text_proj(h)

        # Fusion
        fused = torch.cat([img_emb, txt_emb], dim=1)
        logits = self.fusion_head(fused)
        return logits

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        texts = batch["text"]
        labels = batch["label"]
        valid = batch["label_valid"]

        logits = self(images, texts)
        loss_per_sample = self.criterion(logits, labels)
        # Average loss only over valid labels per study
        loss = (loss_per_sample * valid).sum() / (valid.sum() + 1e-8)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        texts = batch["text"]
        labels = batch["label"]
        valid = batch["label_valid"]

        logits = self(images, texts)
        loss_per_sample = self.criterion(logits, labels)
        loss = (loss_per_sample * valid).sum() / (valid.sum() + 1e-8)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()

        self.val_preds.append((probs, valid_np))
        self.val_targets.append(labels_np)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if len(self.val_targets) == 0:
            return

        y_true = np.concatenate([t for t, _ in zip(self.val_targets, [v for _, v in self.val_preds])], axis=0)
        y_score = np.concatenate([p for p, _ in self.val_preds], axis=0)
        valid_mask = np.concatenate([v for _, v in self.val_preds], axis=0)

        macro_auc = compute_macro_auc(y_true, y_score, valid_mask)
        self.log("val/macro_auc", macro_auc, prog_bar=True)

        self.val_targets = []
        self.val_preds = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def predict_step(self, batch, batch_idx):
        images = batch["image"]
        texts = batch["text"]
        logits = self(images, texts)
        probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy()


class MultimodalKneeDataset(Dataset):
    """Loads preprocessed MRI + tokenized reports.
    
    Handles missing labels by masking them during training.
    """

    def __init__(self, rows: list[dict], n_slices: int = 16, max_text_len: int = 256):
        self.rows = rows
        self.n_slices = n_slices
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]

        # Load image
        npz = np.load(r["path"], allow_pickle=True)
        vol = npz["images"] if "images" in npz else npz[npz.files[0]]
        Z = vol.shape[0]
        if Z >= self.n_slices:
            indices = np.linspace(0, Z - 1, self.n_slices, dtype=int)
        else:
            indices = np.concatenate([np.arange(Z), np.repeat(Z - 1, self.n_slices - Z)])
        slices = vol[indices]
        x_img = np.expand_dims(slices, axis=1)
        x_img = torch.from_numpy(x_img).float()

        # Load text
        text_str = r.get("text_encoded", "")
        tokens = [int(t) for t in str(text_str).strip().split()] if text_str else [1]
        if len(tokens) < self.max_text_len:
            tokens = tokens + [0] * (self.max_text_len - len(tokens))
        else:
            tokens = tokens[: self.max_text_len]
        x_txt = torch.tensor(tokens, dtype=torch.long)

        # Load labels and validity mask
        label_cols = sorted([c for c in r.keys() if c.startswith("label_")])
        if label_cols:
            y = np.array([float(r[c]) if pd.notna(r.get(c)) else 0.0 for c in label_cols], dtype=np.float32)
            valid = np.array([1.0 if pd.notna(r.get(c)) else 0.0 for c in label_cols], dtype=np.float32)
        else:
            y = np.zeros(12, dtype=np.float32)
            valid = np.zeros(12, dtype=np.float32)

        return {
            "image": x_img,
            "text": x_txt,
            "label": torch.from_numpy(y).float(),
            "label_valid": torch.from_numpy(valid).float(),
            "id": r["id"],
        }


def collate_multimodal(batch):
    """Collate function."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    texts = torch.stack([b["text"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    valid = torch.stack([b["label_valid"] for b in batch], dim=0)
    ids = [b["id"] for b in batch]

    return {
        "image": images,
        "text": texts,
        "label": labels,
        "label_valid": valid,
        "id": ids,
    }


def load_manifest(csv_path: Path, fold: int) -> tuple[list[dict], list[dict]]:
    """Load manifest and split by fold."""
    df = pd.read_csv(csv_path)

    train_rows = []
    val_rows = []

    for _, row in df.iterrows():
        row_dict = dict(row)
        if row.get("fold", -1) != fold:
            train_rows.append(row_dict)
        else:
            val_rows.append(row_dict)

    return train_rows, val_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Manifest CSV from build_multimodal_manifest.py")
    parser.add_argument("--fold", type=int, default=0, help="Fold to validate on")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_slices", type=int, default=16)
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--output_dir", default="models/multimodal")
    parser.add_argument("--log_dir", default="runs/multimodal")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # Load data
    train_rows, val_rows = load_manifest(Path(args.csv), args.fold)
    print(f"Fold {args.fold}: {len(train_rows)} train, {len(val_rows)} val")

    train_ds = MultimodalKneeDataset(train_rows, n_slices=args.n_slices, max_text_len=args.max_text_len)
    val_ds = MultimodalKneeDataset(val_rows, n_slices=args.n_slices, max_text_len=args.max_text_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_multimodal
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_multimodal
    )

    # Model
    model = MidFusionImageTextModel(
        n_outputs=12,
        image_backbone="tf_efficientnet_b3",
        embed_dim=256,
        vocab_size=args.vocab_size,
        lr=args.lr,
    )

    # Logger and callbacks
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logger = TensorBoardLogger(save_dir=args.log_dir, name="mid_fusion", version=f"fold{args.fold}")
    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f"multimodal_fold{args.fold}-" + "{epoch}-{val/macro_auc:.4f}",
        monitor="val/macro_auc",
        mode="max",
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor="val/macro_auc", mode="max", patience=10, verbose=True)

    # Train
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint, early_stop],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        log_every_n_steps=10,
        num_sanity_val_steps=2,
    )

    trainer.fit(model, train_loader, val_loader)

    # Save OOF
    best_path = checkpoint.best_model_path
    if best_path:
        print(f"\nBest checkpoint: {best_path}")
        best_model = MidFusionImageTextModel.load_from_checkpoint(best_path)
        preds = trainer.predict(best_model, dataloaders=val_loader)
        all_probs = np.concatenate(preds, axis=0)
        ids = [r["id"] for r in val_rows]

        os.makedirs("preds", exist_ok=True)
        np.save(
            f"preds/multimodal_fold{args.fold}.npy",
            {"ids": ids, "probs": all_probs},
            allow_pickle=True,
        )
        print(f"Saved OOF: preds/multimodal_fold{args.fold}.npy")


if __name__ == "__main__":
    main()
