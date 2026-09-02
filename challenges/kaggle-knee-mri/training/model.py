import torch
import torch.nn as nn
import timm
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, 1)

    def forward(self, x):  # x: (B, S, D)
        w = torch.softmax(self.q(x).squeeze(-1), dim=1)  # (B, S)
        return (x * w.unsqueeze(-1)).sum(1)


def compute_macro_auc(y_true, y_score):
    # y_true, y_score: numpy arrays, shape (N, C)
    try:
        # compute per-class roc and average
        n_classes = y_true.shape[1]
        aucs = []
        for i in range(n_classes):
            try:
                auc = roc_auc_score(y_true[:, i], y_score[:, i])
            except Exception:
                auc = np.nan
            aucs.append(auc)
        # ignore nan
        aucs = [a for a in aucs if not np.isnan(a)]
        if len(aucs) == 0:
            return 0.0
        return float(np.mean(aucs))
    except Exception:
        return 0.0


class SliceLightning(pl.LightningModule):
    def __init__(self, backbone_name='tf_efficientnet_b3', n_slices=16, n_outputs=12, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool='')
        feat_dim = self.backbone.num_features
        self.n_slices = n_slices
        self.pool = AttentionPool(feat_dim)
        self.head = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_outputs))
        self.criterion = nn.BCEWithLogitsLoss()
        # buffers for validation
        self.val_targets = []
        self.val_preds = []

    def forward(self, x):
        # x: (B, S, 1, H, W) -> reshape to (B*S, 3, H, W) or (B*S, 1, H, W)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        # convert to 3 channels if backbone expects 3
        if self.backbone.embeddings is None and self.backbone.num_classes == 0:
            pass
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self.backbone.forward_features(x)
        feats = feats.view(B, S, -1)
        pooled = self.pool(feats)
        logits = self.head(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)
        ids = batch['id']
        logits = self(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        self.val_preds.append(probs)
        self.val_targets.append(labels.detach().cpu().numpy())
        # also log batch loss
        loss = self.criterion(logits, labels)
        self.log('val/loss', loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        if len(self.val_targets) == 0:
            return
        y_true = np.concatenate(self.val_targets, axis=0)
        y_score = np.concatenate(self.val_preds, axis=0)
        macro = compute_macro_auc(y_true, y_score)
        self.log('val/macro_auc', macro, prog_bar=True)
        # clear buffers
        self.val_targets = []
        self.val_preds = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
