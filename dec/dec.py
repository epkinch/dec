import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

config = {
        "lr": 0.001,
        "latent_dim": 10,
        "batch_size": 256,
        "kmeans_seeds": 20,
        "kmeans_iters": 300,
        "n_clusters": 10,
        "batch_size": 256,
        "epochs": 20,
        "alpha": 1.0
    }

# Define model
class StackedAutoEncoder(nn.Module):
    def __init__(self, n_clusters=config['n_clusters'], latent_dim=config['latent_dim'], alpha=config['alpha']):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 500), # Input layer to first hidden layer
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(500, 2000), # Latent representation (bottleneck)
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(2000, config["latent_dim"]) # Deepest layer of encoder
        )
        self.decoder = nn.Sequential(
            nn.Linear(config["latent_dim"], 2000),
            nn.ReLU(True),
            nn.Linear(2000, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 28*28), # Output layer, same size as input
            nn.Sigmoid() # Use Sigmoid to ensure output pixel values are in [0, 1] range
        )
        # Centroids stored directly on the model — initialized later from K-Means
        self.centroids = nn.Parameter(
            torch.randn(n_clusters, latent_dim),
            requires_grad=False  # frozen until Phase 3 begins
        )

    def encode(self, x):
        return self.encoder(self.flatten(x))

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decoder(z)
        return x_recon, z
  
# --- Phase 1: Train the autoencoder (reconstruction only) ---
def train_autoencoder(dataloader, model, loss_fn, optimizer, epochs=20):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch, (X, _) in enumerate(dataloader):          # _ = ignore labels entirely during AE training
            X = X.to(device)
            x_recon, z = model(X)

            # Loss is pixel reconstruction error — shape [batch,784] vs [batch,784]
            loss = loss_fn(x_recon, X.view(X.size(0), -1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch+1:>3d} | Recon Loss: {avg:.6f}")

# --- Extract latent vectors from the entire dataset ---
def get_latent_vectors(dataloader, model):
    model.eval()
    all_z, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            _, z = model(X)
            all_z.append(z.cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_z), np.concatenate(all_labels)

# --- Phase 2: K-Means in latent space ---
def run_kmeans(z, n_clusters=10):
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=20,       # run 20 times with different seeds, keep best
        max_iter=300,
        random_state=42
    )
    cluster_assignments = kmeans.fit_predict(z)
    return cluster_assignments, kmeans

# --- Hungarian matching: align cluster IDs to true class labels ---
def hungarian_accuracy(true_labels, cluster_assignments, n_clusters=10, n_classes=10):
    cost_matrix = np.zeros((n_clusters, n_classes), dtype=np.int64)
    for cluster_id, true_id in zip(cluster_assignments, true_labels):
        cost_matrix[cluster_id, true_id] += 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    remapped = np.array([mapping.get(c, -1) for c in cluster_assignments])
    return accuracy_score(true_labels, remapped), mapping

if __name__ == "__main__":
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="dec/data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="dec/data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'])

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")


    # Create model
    model = StackedAutoEncoder().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ============================================================
    # RUN
    # ============================================================
    print("=== Phase 1: Training Autoencoder ===")
    train_autoencoder(train_dataloader, model, loss_fn, optimizer, epochs=config["epochs"])

    print("\n=== Phase 2: K-Means Clustering in Latent Space ===")
    z_train, labels_train = get_latent_vectors(train_dataloader, model)
    z_test,  labels_test  = get_latent_vectors(test_dataloader,  model)

    print(f"Latent vectors shape: {z_train.shape}")  # should be (60000, LATENT_DIM)

    cluster_assignments_train, kmeans = run_kmeans(z_train, n_clusters=config["n_clusters"])

    acc_train, label_mapping = hungarian_accuracy(labels_train, cluster_assignments_train)
    print(f"Train clustering accuracy: {acc_train*100:.1f}%")
    print(f"Cluster → Digit mapping: {label_mapping}")

    # Apply the same kmeans to test set
    cluster_assignments_test = kmeans.predict(z_test)

    # Remap test clusters using the mapping learned from training
    remapped_test = np.array([label_mapping[c] for c in cluster_assignments_test])
    test_acc = accuracy_score(labels_test, remapped_test)
    print(f"Test  clustering accuracy: {test_acc*100:.1f}%")

