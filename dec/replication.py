from clustpy.data import create_subspace_data
from clustpy.deep import DEC
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import numpy as np


data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
dec = DEC(n_clusters=3, pretrain_epochs=3, clustering_epochs=3)
dec.fit(data)


predicted_labels = dec.labels_

nmi = normalized_mutual_info_score(labels, predicted_labels)
ari = adjusted_rand_score(labels, predicted_labels)

def clustering_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n_classes = max(y_true.max(), y_pred.max()) + 1
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    return confusion[row_ind, col_ind].sum() / len(y_true)

acc = clustering_accuracy(labels, predicted_labels)

print(f"ACC: {acc:.4f}")
print(f"NMI: {nmi:.4f}")
print(f"ARI: {ari:.4f}")