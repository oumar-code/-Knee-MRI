#!/usr/bin/env python3
"""Build the study-level manifest used by the Knee MRI pipeline.

Kaggle inputs:
  train.csv: StudyInstanceUID, Report, and the twelve target columns
  test.csv: StudyInstanceUID and Report (Report is absent at scoring time)
  train_series.csv/test_series.csv: series descriptors

The output uses the repository's internal schema:
  id, path, report, text_encoded, fold, label_0 ... label_11

Important: reports are available for training studies but are not provided for
Kaggle test studies. Therefore report features must not be required at test
inference. The resulting manifest is useful for experiments, teacher models,
or report ablations; the production leaderboard model should use image and
available test-time metadata only unless a separate text-imputation strategy
is validated without leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.text_tokenizer import SimpleTokenizer

LABEL_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, help="Kaggle train.csv")
    p.add_argument("--npz-root", required=True, help="Directory containing {StudyInstanceUID}.npz")
    p.add_argument("--out", required=True, help="Output manifest CSV")
    p.add_argument("--tokenizer-out", default="", help="Optional tokenizer JSON output")
    p.add_argument("--max-text-len", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument("--min-freq", type=int, default=1)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-missing-images", action="store_true")
    return p.parse_args()


def make_folds(ids: pd.Series, n_folds: int, seed: int) -> pd.Series:
    if n_folds < 2:
        return pd.Series(np.zeros(len(ids), dtype=np.int64), index=ids.index)
    if len(ids) < n_folds:
        raise ValueError(f"{len(ids)} studies cannot be split into {n_folds} folds")
    folds = np.full(len(ids), -1, dtype=np.int64)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(ids)):
        folds[valid_idx] = fold
    return pd.Series(folds, index=ids.index)


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels)
    npz_root = Path(args.npz_root)
    out_path = Path(args.out)

    source = pd.read_csv(labels_path)
    required = {"StudyInstanceUID", "Report", *LABEL_COLUMNS}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"{labels_path} is missing columns: {sorted(missing)}")

    source = source.copy()
    source["StudyInstanceUID"] = source["StudyInstanceUID"].astype(str)
    source["Report"] = source["Report"].fillna("").astype(str)

    # Keep only studies for which preprocessing produced a volume. This avoids
    # silently creating unusable training rows.
    source["path"] = source["StudyInstanceUID"].map(
        lambda uid: str((npz_root / f"{uid}.npz").resolve())
    )
    exists = source["path"].map(lambda p: Path(p).exists())
    if not args.allow_missing_images and not bool(exists.all()):
        missing_ids = source.loc[~exists, "StudyInstanceUID"].head(10).tolist()
        raise FileNotFoundError(
            f"{int((~exists).sum())} studies have no preprocessed .npz. "
            f"Examples: {missing_ids}. Run preprocess.py first or pass --allow-missing-images."
        )
    source = source.loc[exists].reset_index(drop=True)
    if source.empty:
        raise ValueError("No rows remain after matching train.csv to .npz files")

    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size, min_freq=args.min_freq)
    tokenizer.fit(source["Report"].tolist())

    output = pd.DataFrame({
        "id": source["StudyInstanceUID"],
        "path": source["path"],
        "report": source["Report"],
        "text_encoded": source["Report"].map(
            lambda text: " ".join(map(str, tokenizer.encode(text, args.max_text_len)))
        ),
    })
    output["fold"] = make_folds(output["id"], args.folds, args.seed).to_numpy()
    for idx, column in enumerate(LABEL_COLUMNS):
        # Preserve missing labels as NaN; the training loader should not treat
        # an unlabeled study as a negative example.
        output[f"label_{idx}"] = pd.to_numeric(source[column], errors="coerce").to_numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    if args.tokenizer_out:
        tokenizer_path = Path(args.tokenizer_out)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path.write_text(json.dumps({
            "vocab_size": args.vocab_size,
            "min_freq": args.min_freq,
            "word2idx": tokenizer.word2idx,
            "idx2word": tokenizer.idx2word,
            "max_text_len": args.max_text_len,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(output)} studies to {out_path}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print("Labels: " + ", ".join(f"label_{i}={name}" for i, name in enumerate(LABEL_COLUMNS)))
    print("Note: Kaggle test reports are unavailable; do not require text at test inference.")


if __name__ == "__main__":
    main()
