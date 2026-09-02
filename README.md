# Kaggle — Knee MRI Abnormality Detection

Standardized README for the repository with quickstart and explanations.

## Quickstart

1. Install dependencies:

```bash
pip install -r challenges/kaggle-knee-mri/requirements.txt
pip install -U pip
pip install timm pytorch-lightning transformers
```

2. Preprocess DICOMs into per-study .npz (saves pixel_spacing and slice_thickness):

```bash
python challenges/kaggle-knee-mri/preprocess.py --dicom-root /path/to/dicom --out-dir data/preprocessed
```

3. Generate dataset CSV:

```bash
python challenges/kaggle-knee-mri/scripts/generate_csv.py --npz_root data/preprocessed --out challenges/kaggle-knee-mri/data/dataset.csv
```

4. Run 5-fold training (uses PyTorch Lightning, single best checkpoint per fold):

```bash
python challenges/kaggle-knee-mri/scripts/run_folds.py --csv challenges/kaggle-knee-mri/data/dataset.csv --folds 5
```

5. Collect OOF predictions:

```bash
python challenges/kaggle-knee-mri/scripts/collect_oof.py --preds_dir preds --out challenges/kaggle-knee-mri/preds_oof.csv
```

6. View TensorBoard:

```bash
tensorboard --logdir runs
```

## What's included

- Updated preprocessing (ImagePositionPatient ordering + per-study metadata)
- PyTorch Lightning training module with EarlyStopping and TensorBoard logging
- Runner script to train all folds and collect OOF predictions
- Mid-fusion multimodal notebook (image + text fusion example)

## Notes

- Training saves a single best checkpoint per fold (monitored by val/macro_auc).
- The preprocessing outputs are compressed .npz files containing: images (Z,H,W), pixel_spacing, slice_thickness, image_positions.
