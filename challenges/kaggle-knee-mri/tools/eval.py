import numpy as np
from sklearn.metrics import roc_auc_score


def per_class_auc(y_true, y_pred):
    aucs = []
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except Exception:
            auc = float('nan')
        aucs.append(auc)
    return aucs


if __name__ == '__main__':
    # example usage: python eval.py oof.npz
    import sys
    arr = np.load(sys.argv[1])
    preds = arr['preds']
    gts = arr['gts']
    aucs = per_class_auc(gts, preds)
    print('Per-class AUCs:')
    for i, a in enumerate(aucs):
        print(i, a)
    print('Macro AUC:', np.nanmean(aucs))
