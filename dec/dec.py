"""TODO weird bug where after cluster refinement the cluster assignment is better but this causes the accuracy of the test set to go down"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
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
        "epochs": 10,
        "alpha": 1.0,
        "refine_epochs":10,
        "tol": 0.001
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
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 2000), # Latent representation (bottleneck)
            nn.ReLU(True),
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
            requires_grad=False
        )

    def encode(self, x):
        return self.encoder(self.flatten(x))
    
    def soft_assign(self, z):
        dist = torch.cdist(z, self.centroids) ** 2
        num = (1+dist/self.alpha) ** (-(self.alpha+1)/2)
        return num/num.sum(dim=1, keepdim=True)
    
    def init_centroids(self, cluster_centers):
        with torch.no_grad():
            self.centroids.copy_(
                torch.tensor(cluster_centers, dtype=torch.float32)
            )
        self.centroids.requires_grad = True

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

# --- Phase 3: Refine cluster centroids
def target_distribution(q):
    f = q.sum(dim=0)
    p = (q ** 2) / f
    return p / p.sum(dim=1, keepdim=True)

def train_dec(dataloader, model, optimizer_dec, tol=config['tol'], epochs=20, run = "epoch"):
    prev_assignments = None
    epoch=0
    if run == "tol":
        print(f"\nRunning until only {tol*100}% have changed")
    elif run == "epoch":
        print(f"\nRunning for {epochs} epochs")

    while True:
        if run == "epoch" and epoch >= epochs: break
        # Compute P over full dataset
        model.eval()
        all_q = []
        with torch.no_grad():
            for X, _ in dataloader:
                z = model.encode(X.to(device))
                all_q.append(model.soft_assign(z).cpu())
        all_q  = torch.cat(all_q)
        p_full = target_distribution(all_q)

        # Convergence check
        current_assignments = all_q.argmax(dim=1).numpy()
        if prev_assignments is not None:
            changed = (current_assignments != prev_assignments).mean()
            print(f"  Epoch {epoch+1}: {changed*100:.2f}% changed")
            if changed < tol:
                print(f"Converged at epoch {epoch+1}"); break
        prev_assignments = current_assignments.copy()

        # Training pass
        model.train()
        total_loss = 0
        for batch_idx, (X, _) in enumerate(dataloader):
            X = X.to(device)
            start   = batch_idx * dataloader.batch_size
            p_batch = p_full[start : start + len(X)].to(device)

            z       = model.encode(X)
            q_batch = model.soft_assign(z)
            loss    = F.kl_div(q_batch.log(), p_batch, reduction='batchmean')

            optimizer_dec.zero_grad()
            loss.backward()
            optimizer_dec.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:>3d} | KL Loss: {total_loss/len(dataloader):.6f}")
        epoch+=1

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

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=config['batch_size'], shuffle=True)
    train_dataloader_noshuffle = DataLoader(training_data, batch_size=config['batch_size'], shuffle=False)
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

    # Train model - create a latent space via encoder
    print("=== Phase 1: Training Autoencoder ===")
    train_autoencoder(train_dataloader, model, loss_fn, optimizer, epochs=config["epochs"])

    # Create intial k-means centroids in latent space
    print("\n=== Phase 2: K-Means Clustering in Latent Space ===")
    z_train, labels_train = get_latent_vectors(train_dataloader_noshuffle, model)
    z_test,  labels_test  = get_latent_vectors(test_dataloader,  model)
    print(f"Latent vectors shape: {z_train.shape}")  # should be (60000, latent_dim)
    cluster_assignments_train, kmeans = run_kmeans(z_train, n_clusters=config["n_clusters"])
    cluster_assignments_test, _ = run_kmeans(z_test, n_clusters=config["n_clusters"])

    print("\n=== Phase 2b: Initial Accuracy")
    acc_train, label_mapping = hungarian_accuracy(labels_train, cluster_assignments_train)
    acc_test, _ = hungarian_accuracy(labels_test, cluster_assignments_test)
    print(f"Train clustering accuracy: {acc_train*100:.1f}%")
    print(f"Test clustering accuracy: {acc_test*100:.1f}%")
    print(f"Cluster → Digit mapping: {label_mapping}")

    print("\n=== Phase 3: Cluster refinement ===")
    model.init_centroids(kmeans.cluster_centers_)
    optimizer_dec = torch.optim.SGD(
        model.parameters(),  # encoder + centroids, decoder gets frozen naturally
        lr=0.01
    )
    train_dec(train_dataloader_noshuffle, model, optimizer_dec, epochs=config["refine_epochs"], run="epoch") # run = epoch / tol

    print("\n=== Final Evaluation ===")
    model.eval()
    all_q, all_labels = [], []
    with torch.no_grad():
        for X, y in test_dataloader:
            z = model.encode(X.to(device))
            q = model.soft_assign(z)
            all_q.append(q.cpu())
            all_labels.append(y)

    all_q = torch.cat(all_q)
    all_labels = torch.cat(all_labels).numpy()
    final_assignments = all_q.argmax(dim=1).numpy()

    test_acc, label_mapping = hungarian_accuracy(all_labels, final_assignments)
    print(f"Test clustering accuracy: {test_acc*100:.1f}%")
    print(f"Cluster -> Digit mapping: {label_mapping}")