Kaggle — Knee MRI Abnormality Detection
This folder contains planning, notes, and starter code for the Kaggle Knee MRI competition: develop multimodal models (MRI images + original radiology reports) to detect 12 clinically important knee abnormalities. The goal is to maximize the macro-averaged ROC-AUC across the 12 targets and to produce a reproducible training + inference pipeline and submission.

Quick links
Competition: Kaggle (Knee MRI Abnormality Detection)
Repository path: challenges/kaggle-knee-mri/
Maintainer: @oumar-code
Challenge summary
Inputs: MRI studies (DICOM series) paired with radiology reports.

Targets (12): ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's cyst, Contusion, Fracture.

Evaluation: Macro-averaged AUC-ROC (mean of 12 per-class AUCs).

Submission format: CSV with header

StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture

Objective
Build a robust, reproducible multimodal pipeline that:

Produces a strong baseline quickly (slice-wise 2D CNN + study-level pooling).
Leverages report text for complementary signals (Clinical BERT variants).
Iterates toward stronger volume-aware models, advanced fusion, and diverse ensembling to maximize macro AUC.
High-level strategy (two-track)
Baseline track (fast, reliable): 2D slice-wise ImageNet pretrained backbone (EfficientNet/ResNet) → per-slice features → study-level aggregation (mean/attention) → MLP head for 12 outputs.
Multimodal track (parallel): text-only baseline (Bio/Clinical BERT) → late/mid fusion of image + report embeddings (concatenate or cross-attention). Progressively increase model sophistication (3D/2.5D, slice transformers, self-supervised pretraining).
Use group-aware cross-validation (StudyInstanceUID or PatientID) and macro AUC as the main selection metric. Ensembling/TTA/pseudo-labeling for final boosts.

3-week sprint timeline (recommended)
Day 0–2: EDA, DICOM tooling, preprocess & caching, class distribution analysis.
Day 2–6: Baseline 2D slice model, 5-fold CV, first leaderboard submission.
Day 4–10: Report-only and simple multimodal fusion (concat), submit improvements.
Day 7–14: Volume-aware experiments (2.5D/3D, slice transformer), SSL pretraining if time.
Day 14–21: Ensembling, TTA, pseudo-labeling, final submissions and write-up.
Actionable checklist (priority order)
 Create dataset ingestion & caching utilities (DICOM -> stacked numpy / compressed tensors).
 EDA notebook: series types, counts, slice distributions, label prevalence, example studies.
 Baseline: slice-wise 2D CNN + study aggregation + train/infer scripts + 5-fold CV.
 Evaluation tools: per-class ROC-AUC, macro AUC, PR-AUC, calibration plots.
 Report-only model: preprocess, fine-tune BioClinicalBERT/PubMedBERT, get predictions.
 Multimodal fusion: late ensemble, then mid/ cross-attention fusion.
 Advanced models: 2.5D/3D, slice transformer, MAE-style pretraining.
 Ensembling & TTA: diverse models, stacking, test-time augmentation.
 Pseudo-labeling (optional & careful): high-confidence predictions used to augment training.
 Inference scripts & submission exporter (correct CSV format).
 Documentation: this README, experiment log, runbook for submission reproducibility.
Dataset & EDA checklist
Count studies and unique patients.
List series by SeriesDescription/SequenceName (e.g., Sagittal PD, Coronal, Axial) and distribution.
Inspect slice counts per series and spacing (pixel spacing, slice thickness).
Visualize representative slices per label and failure modes.
Check for PHI in reports and strip if necessary.
Preprocessing recommendations
DICOM reading: pydicom + fastnp/torch I/O. Cache preprocessed volumes as compressed .npz or .npy to speed training.
Intensity normalization: per-volume z-score or percentile-based scaling (0.5th–99.5th). Avoid CT windowing rules.
Resize to a consistent in-plane resolution (e.g., 256x256 or 320x320). Preserve aspect ratio or center crop when appropriate.
Slice sampling: uniform sampling (N=16 or 32) for baseline; central cropping of key anatomy.
Augmentations: flips, small rotations, affine, intensity jitter. Medical plausibility: careful with left-right flips if laterality matters.
Baseline code snippets
These snippets are intended as starting points — adapt to style and frameworks you prefer.

DICOM loader (simplified)
# challenges/kaggle-knee-mri/data/dicom_loader.py
import pydicom
import numpy as np
from pathlib import Path

def load_series(series_folder: Path):
    files = sorted(series_folder.glob('*.dcm'))
    slices = [pydicom.dcmread(str(p)).pixel_array for p in files]
    volume = np.stack(slices, axis=0).astype('float32')
    return volume
Slice dataset & sampler (PyTorch)
# challenges/kaggle-knee-mri/data/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class StudySliceDataset(Dataset):
    def __init__(self, study_paths, labels, n_slices=16, transform=None):
        self.study_paths = study_paths
        self.labels = labels
        self.n_slices = n_slices
        self.transform = transform

    def __len__(self):
        return len(self.study_paths)

    def __getitem__(self, idx):
        vol = np.load(self.study_paths[idx])['arr_0']  # preprocessed npz
        L = vol.shape[0]
        # uniform sampling
        if L >= self.n_slices:
            indices = np.linspace(0, L - 1, self.n_slices, dtype=int)
        else:
            indices = np.concatenate([np.arange(L), np.repeat(L-1, self.n_slices-L)])
        slices = vol[indices]
        # stack as channels
        x = np.stack([s for s in slices], axis=0)
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y).float()
Simple model: EfficientNet + attention pooling
# challenges/kaggle-knee-mri/models/baseline.py
import timm
import torch
import torch.nn as nn

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, 1)

    def forward(self, x):  # x: (B, S, D)
        w = torch.softmax(self.q(x).squeeze(-1), dim=1)  # (B, S)
        return (x * w.unsqueeze(-1)).sum(1)

class SliceModel(nn.Module):
    def __init__(self, backbone_name='tf_efficientnet_b3', pretrained=True, n_outputs=12):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='')
        feat_dim = self.backbone.num_features
        self.pool = AttentionPool(feat_dim)
        self.head = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_outputs))

    def forward(self, x):
        # x: (B, S, C, H, W) -> treat as batch of (B*S, C, H, W)
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        f = self.backbone.forward_features(x)  # (B*S, D, 1, 1) or (B*S, D)
        f = f.view(B, S, -1)
        s = self.pool(f)
        out = self.head(s)
        return out
Training notes
Loss: BCEWithLogitsLoss per target. Monitor per-class ROC-AUC and macro AUC.
Optimizer: AdamW, lr 1e-4, weight_decay 1e-2. Use OneCycleLR or cosine schedule.
Mixed precision training (apex or native AMP) and gradient accumulation if needed.
Multimodal plan (detailed)
Text pipeline
Use BioClinicalBERT / PubMedBERT / ClinicalBERT if available; otherwise RoBERTa/BERT base.
Preprocess reports (tokenize, remove PHI, keep relevant sections). Truncate/pad to 512 tokens.
Extract [CLS] embeddings or fine-tune end-to-end with a linear head for 12 outputs.
Fusion approaches
Late fusion: average or stack image-only and text-only model probabilities.
Mid fusion: concatenate image study embedding + text CLS embedding → MLP → 12 outputs.
Cross-attention: multimodal transformer where text tokens attend to slice tokens (best performance but heavier).
Training strategy
Train image-only and text-only baselines first (stable OOF preds).
For multimodal, start with frozen backbones and train fusion head; progressively unfreeze.
Use group-KFold OOF predictions to train a stacking model for final blending.
Validation & CV
Use GroupKFold by StudyInstanceUID or PatientID to prevent leakage.
Primary early-stopping metric: macro ROC-AUC. Log per-class AUCs to target weak classes.
Keep a local holdout (not used in CV) for final validation to reduce leaderboard overfitting risk.
Ensembling & final tuning
Model diversity: vary backbones, input sizes, slice counts, and modalities (image, text, multimodal).
TTA: flips, small rotations, intensity jitter — average predictions.
Stacking: small meta-model (LogisticRegression / LightGBM) on OOF predictions.
Pseudo-labeling: only use high-confidence predictions with conservative thresholds.
Repo layout (suggested)
challenges/kaggle-knee-mri/

README.md ← this file
data/
dicom_loader.py
preprocess.py
dataset.py
models/
baseline.py
multimodal.py
slice_transformer.py
training/
train.py
infer.py
eval.py
notebooks/
EDA.ipynb
baseline_train.ipynb
submissions/
sample_submission.csv
experiments/
logs/
checkpoints/
Reproducibility & notes
Fix random seeds for numpy/torch/python random.
Log environment, library versions, and command-line args in experiment logs.
Use mixed precision and caching to speed development.
Next steps
Create the challenges/kaggle-knee-mri/ folder with this README (done).
Scaffold starter files: data/dicom_loader.py, data/dataset.py, models/baseline.py, training/train.py, training/infer.py, notebooks/EDA.ipynb.
Implement the runnable baseline (DICOM -> preprocessed npz -> slice-wise EfficientNet -> attention pooling -> train with 5-fold CV) and push to a branch.
