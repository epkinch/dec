import csv
import sys

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from clustpy.data import load_reuters


# --- Hungarian matching: align cluster IDs to true class labels ---
def hungarian_accuracy(true_labels, cluster_assignments, n_clusters=10, n_classes=10):
    cost_matrix = np.zeros((n_clusters, n_classes), dtype=np.int64)
    for cluster_id, true_id in zip(cluster_assignments, true_labels):
        cost_matrix[cluster_id, true_id] += 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    remapped = np.array([mapping.get(c, -1) for c in cluster_assignments])
    return accuracy_score(true_labels, remapped), mapping

def save_accuracy(model, dataset_name, hyperparameter, test_acc, filename="accuracies.csv"):
    header = ["model", "dataset", "hyperparameter", "test_acc"]
    row = [model, dataset_name, hyperparameter, test_acc]

    # Append if file exists, otherwise write header
    try:
        with open(filename, "x", newline="") as f:  # creates file if not exists
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)
    except FileExistsError:
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

def main():
    # MNIST
    training_data = datasets.MNIST(
        root="dec/data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    X_train = training_data.data.numpy().reshape(-1, 28*28)
    y_train = training_data.targets.numpy()

    test_data = datasets.MNIST(
        root="dec/data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    X_test = test_data.data.numpy().reshape(-1, 28*28)
    y_test = test_data.targets.numpy()

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=10, n_init="auto").fit(X_train)
        assignments = kmeans.predict(X_test)
        acc, _ = hungarian_accuracy(y_test, assignments)
        save_accuracy("KMeans", "MNIST", i, acc, "replication_accuracies.csv")

    X_train_scaled = X_train / 255.0
    X_pca = PCA(n_components=50).fit_transform(X_train_scaled)

    for n_neighbors in range(2, 12):
        sc = SpectralClustering(
            n_clusters=10,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors,
            assign_labels='kmeans',
            n_init=5,
            n_jobs=-1
        )
        assignments = sc.fit_predict(X_pca)
        acc, _ = hungarian_accuracy(y_train, assignments)
        save_accuracy("Spectral Clustering", "MNIST", n_neighbors, acc, "replication_accuracies.csv")

    # STL
        
    
    # REUTERS-10K
    
        

if __name__ == '__main__':
    main()