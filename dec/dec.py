import csv
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from clustpy.data import load_reuters

config = {
        "lr": 0.01,
        "input_dim": 28*28,
        "latent_dim": 10,
        "batch_size": 256,
        "kmeans_seeds": 30,
        "kmeans_iters": 300,
        "n_clusters": 10,
        "batch_size": 256,
        "epochs": 2,
        "alpha": 1.0,
        "refine_epochs": 2,
        "tol": 0.001
    }

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"

# Define model
class StackedAutoEncoder(nn.Module):
    def __init__(self, n_clusters=config['n_clusters'], latent_dim=config['latent_dim'], alpha=config['alpha']):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(config['input_dim'], 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 2000),
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
            nn.Linear(500, config['input_dim']),
            nn.Sigmoid()
        )
        # Centroids initialized later from K-Means
        self.centroids = nn.Parameter(
            torch.randn(n_clusters, config["latent_dim"]),
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
        for batch, (X, _) in enumerate(dataloader):
            X = X.to(device)
            x_recon, z = model(X)

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
    p = (q ** 2) / (f + 1e-10)
    return p / p.sum(dim=1, keepdim=True)

def train_dec(dataloader, model, optimizer_dec, tol=config['tol'], epochs=20, run = "epoch", backprop = True):
    prev_assignments = None

    # Freeze decoder — KL loss should only update encoder + centroids
    for param in model.decoder.parameters():
        param.requires_grad = False

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
        all_q = torch.cat(all_q)
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
        p_loader = DataLoader(
            torch.utils.data.TensorDataset(p_full),
            batch_size=dataloader.batch_size,
            shuffle=False  # must match dataloader ordering
        )
        model.train()
        total_loss = 0
        for (X, _), (p_batch,) in zip(dataloader, p_loader):
            X       = X.to(device)
            p_batch = p_batch.to(device)

            z = model.encode(X)
            q_batch = model.soft_assign(z)
            # loss    = F.kl_div(q_batch.log(), p_batch, reduction='batchmean')
            loss = (p_batch * ((p_batch + 1e-10) / (q_batch + 1e-10)).log()).sum(dim=1).mean()
            
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

def run_pipeline(dataset_config, hyperparameter, backprop):
    # update config
    training_data = dataset_config['train']
    test_data = dataset_config['test']
    if callable(training_data):
        training_data = training_data()
    if callable(test_data):
        test_data = test_data()

    config['alpha'] = hyperparameter
    if dataset_config['name'] == "MNIST":
        config['input_dim'] = 28*28
   
    
    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=config['batch_size'], shuffle=True)
    train_dataloader_noshuffle = DataLoader(training_data, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'])

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
    cluster_assignments_test = kmeans.predict(z_test)

    print("\n=== Phase 2b: Initial Accuracy")
    acc_train, label_mapping = hungarian_accuracy(labels_train, cluster_assignments_train)
    acc_test, _ = hungarian_accuracy(labels_test, cluster_assignments_test)
    print(f"Train clustering accuracy: {acc_train*100:.1f}%")
    print(f"Test clustering accuracy: {acc_test*100:.1f}%")
    print(f"Cluster → Digit mapping: {label_mapping}")

    if not backprop:
        return (acc_test + acc_train)/2*100
 
    print("\n=== Phase 3: Cluster refinement ===")
    
    model.init_centroids(kmeans.cluster_centers_)
    optimizer_dec = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),  # encoder + centroids, decoder gets frozen naturally
        lr=config["lr"]
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
    print(f"Model: {dataset_config['name']}")
    print(f"Latent Dimensions: {config['latent_dim']}")
    print(f"Test clustering accuracy: {test_acc*100:.1f}%")
    return test_acc*100
   
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

def load_reuters_10k(train):
    if train:
        training_data = TensorDataset(
            torch.tensor(load_reuters("train", return_X_y=True)[0], dtype=torch.float32),
            torch.tensor(load_reuters("train", return_X_y=True)[1], dtype=torch.long)
        )
    else:
        training_data = TensorDataset(
            torch.tensor(load_reuters("test", return_X_y=True)[0], dtype=torch.float32),
            torch.tensor(load_reuters("test", return_X_y=True)[1], dtype=torch.long)
        )

    N = len(training_data)
    subset_size = 10000
    indices = torch.randperm(N)[:subset_size]
    X_sub = training_data.tensors[0][indices]
    y_sub = training_data.tensors[1][indices]
    training_data = TensorDataset(X_sub, y_sub)
    return training_data

def main():
    dataset_config = {
        "name": "MNIST",
        "train": datasets.MNIST(
            root="dec/data",
            train=True,
            download=True,
            transform=ToTensor(),
        ),
        "test": datasets.MNIST(
            root="dec/data",
            train=False,
            download=True,
            transform=ToTensor(),
        ),
        "n_clusters": 10,
    }
    if len(sys.argv) > 1:
        if sys.argv[1] == "1":
            dataset_configs = [
                {
                    "name": "MNIST",
                    "train": datasets.MNIST(
                        root="dec/data",
                        train=True,
                        download=True,
                        transform=ToTensor()
                    ),
                    "test": datasets.MNIST(
                        root="dec/data",
                        train=False,
                        download=True,
                        transform=ToTensor()
                    ),
                    "n_clusters": 10
                },
                # {
                #     "name": "STL10",
                #     "train": datasets.STL10(
                #         root="dec/data",
                #         split="train",
                #         download=True,
                #         transform=ToTensor()
                #     ),
                #     "test": datasets.STL10(
                #         root="dec/data",
                #         split="test",
                #         download=True,
                #         transform=ToTensor()
                #     ),
                #     "n_clusters": 10
                # },
                # {
                #     "name": "REUTERS10k",
                #     "train": lambda: load_reuters_10k(True),
                #     "test": lambda: load_reuters_10k(False),
                #     "n_clusters": 4,
                # },
            ]
            for dataset_config in dataset_configs:
                backprop = True
                for hyperparameter in range(1, 11):
                    acc = run_pipeline(dataset_config, hyperparameter, backprop)
                    save_accuracy(backprop, dataset_config['name'], hyperparameter, acc)

                    
            return
    else:
        run_pipeline(dataset_config, 1, True)

if __name__ == "__main__":
    main()