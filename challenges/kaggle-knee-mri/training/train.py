import argparse
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from training.datamodule import KneeDataModule
from training.model import SliceLightning
import pandas as pd


def load_csv_rows(csv_path, fold):
    df = pd.read_csv(csv_path)
    # expected columns: id, path, fold (0..F-1), label_0..label_11 (optional)
    rows = []
    for _, r in df.iterrows():
        labels = None
        label_cols = [c for c in df.columns if c.startswith('label_')]
        if len(label_cols) > 0:
            labels = r[label_cols].values.astype(float)
        split = 'train' if r.get('fold', -1) != fold else 'val'
        rows.append({'id': r['id'], 'path': r['path'], 'labels': labels, 'split': split})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_slices', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--model', type=str, default='tf_efficientnet_b3')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--checkpoint_dir', type=str, default='models')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    rows = load_csv_rows(args.csv, args.fold)
    dm = KneeDataModule(rows, batch_size=args.batch_size, n_slices=args.n_slices)
    dm.setup()

    model = SliceLightning(backbone_name=args.model, n_slices=args.n_slices)

    logger = TensorBoardLogger(save_dir=args.log_dir, name=f'fold{args.fold}')
    checkpoint = ModelCheckpoint(dirpath=args.checkpoint_dir, filename=f'fold{args.fold}-' + '{epoch}-{val/macro_auc:.4f}', monitor='val/macro_auc', mode='max', save_top_k=1)
    early = EarlyStopping(monitor='val/macro_auc', mode='max', patience=7)

    trainer = Trainer(logger=logger, callbacks=[checkpoint, early], max_epochs=args.max_epochs, gpus=args.gpus if args.gpus>0 else None)
    trainer.fit(model, dm)

    # After training, save oof preds on val set
    ckpt_path = checkpoint.best_model_path
    print('Best checkpoint:', ckpt_path)
    # Load best model and run prediction on val set
    if ckpt_path:
        best = SliceLightning.load_from_checkpoint(ckpt_path)
        best.eval()
        preds = trainer.predict(best, dataloaders=dm.predict_dataloader())
        # preds is list of numpy arrays per batch
        import numpy as np
        all_probs = np.concatenate(preds, axis=0)
        ids = [r['id'] for r in rows if r['split']=='val']
        out = {'ids': ids, 'probs': all_probs}
        np.save(os.path.join('preds', f'fold{args.fold}.npy'), out)

if __name__ == '__main__':
    main()
