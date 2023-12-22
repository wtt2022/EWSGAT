from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.metrics import f1_score


def AC(pred, labels):
    pred = np.array(pred)
    labels = np.array(labels)
    labels = labels.astype(np.int64)
    assert pred.size == labels.size
    D = max(pred.max(), labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(pred.size):
        w[pred[i], labels[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / pred.size

def F1(pred, labels):
    return f1_score(labels, pred, average='weighted')

