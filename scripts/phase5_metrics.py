"""Rows are true labels and columns are predictions, ordered N, V."""
import numpy as np


def metrics(matrix):
    c = np.asarray(matrix)
    support = c.sum(axis=1)
    predicted = c.sum(axis=0)
    tp = c.diagonal()
    precision = tp / np.maximum(predicted, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * tp / np.maximum(predicted + support, 1)
    return dict(accuracy=float(tp.sum() / c.sum()), macro_f1=float(f1.mean()),
                confusion_matrix=c.tolist(), classes={
                    label: dict(precision=float(precision[i]), recall=float(recall[i]),
                                f1=float(f1[i]), support=int(support[i]))
                    for i, label in enumerate(("N", "V"))})
