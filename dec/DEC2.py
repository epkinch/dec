import os
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================


desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

config = {
    "lr": 0.01,
    "latent_dim": 10,
    "batch_size": 256,
    "kmeans_seeds": 30,
    "kmeans_iters": 300,
    "n_clusters": 10,
    "epochs": 75,
    "alpha": 1.0,
    "refine_epochs": 50,
    "tol": 0.001,
    "tsne_samples": 2000,
    "snapshot_epochs": [0, 3, 6, 9, 12],
    "figure_dir": os.path.join(desktop_path, "dec_fig5_outputs")
}

os.makedirs(config["figure_dir"], exist_ok=True)

print("Saving plots to:", config["figure_dir"])

# ============================================================
# MODEL
# ============================================================
class StackedAutoEncoder(nn.Module):
    def __init__(self, n_clusters=config['n_clusters'], latent_dim=config['latent_dim'], alpha=config['alpha']):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.flatten = nn.Flatten()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 2000),
            nn.ReLU(True),
            nn.Linear(2000, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 28 * 28),
            nn.Sigmoid()
        )

        self.centroids = nn.Parameter(
            torch.randn(n_clusters, latent_dim),
            requires_grad=False
        )

    def encode(self, x):
        x = self.flatten(x)
        return self.encoder(x)

    def soft_assign(self, z):
        dist = torch.cdist(z, self.centroids) ** 2
        num = (1 + dist / self.alpha) ** (-(self.alpha + 1) / 2)
        return num / num.sum(dim=1, keepdim=True)

    def init_centroids(self, cluster_centers):
        with torch.no_grad():
            self.centroids.copy_(torch.tensor(cluster_centers, dtype=torch.float32, device=self.centroids.device))
        self.centroids.requires_grad = True

    def forward(self, x):
        x_flat = self.flatten(x)
        z = self.encoder(x_flat)
        x_recon = self.decoder(z)
        return x_recon, z


# ============================================================
# TRAIN AUTOENCODER
# ============================================================
def train_autoencoder(dataloader, model, loss_fn, optimizer, epochs=20):
    size = len(dataloader.dataset)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch, (X, _) in enumerate(dataloader):
            X = X.to(device)
            x_recon, _ = model(X)

            loss = loss_fn(x_recon, X.view(X.size(0), -1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch % 100 == 0:
                current = (batch + 1) * len(X)
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch+1:>3d} | Recon Loss: {avg:.6f}")


# ============================================================
# LATENT EXTRACTION
# ============================================================
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


def get_latent_subset(dataloader, model, max_samples=2000):
    model.eval()
    all_z, all_labels = [], []
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            z = model.encode(X).cpu().numpy()

            all_z.append(z)
            all_labels.append(y.numpy())
            total += len(X)

            if total >= max_samples:
                break

    Z = np.concatenate(all_z, axis=0)[:max_samples]
    Y = np.concatenate(all_labels, axis=0)[:max_samples]
    return Z, Y


# ============================================================
# KMEANS
# ============================================================
def run_kmeans(z, n_clusters=10):
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=20,
        max_iter=300,
        random_state=42
    )
    cluster_assignments = kmeans.fit_predict(z)
    return cluster_assignments, kmeans


# ============================================================
# DEC UTILITIES
# ============================================================
def target_distribution(q):
    f = q.sum(dim=0)
    p = (q ** 2) / (f + 1e-10)
    return p / p.sum(dim=1, keepdim=True)


def hungarian_accuracy(true_labels, cluster_assignments, n_clusters=10, n_classes=10):
    cost_matrix = np.zeros((n_clusters, n_classes), dtype=np.int64)

    for cluster_id, true_id in zip(cluster_assignments, true_labels):
        cost_matrix[cluster_id, true_id] += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    remapped = np.array([mapping.get(c, -1) for c in cluster_assignments])

    return accuracy_score(true_labels, remapped), mapping


def evaluate_dec_clustering(dataloader, model):
    model.eval()
    all_q, all_labels = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            z = model.encode(X)
            q = model.soft_assign(z)
            all_q.append(q.cpu())
            all_labels.append(y)

    all_q = torch.cat(all_q)
    all_labels = torch.cat(all_labels).numpy()
    assignments = all_q.argmax(dim=1).numpy()

    acc, mapping = hungarian_accuracy(all_labels, assignments)
    return acc, assignments, all_labels, mapping


# ============================================================
# PLOTTING
# ============================================================
def plot_tsne_epoch(latent_vectors, labels, epoch, save_dir):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42
    )
    z_2d = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=8,
        alpha=0.8
    )
    plt.title(f"t-SNE of Latent Representation at Epoch {epoch}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, ticks=range(10), label="Digit Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"tsne_epoch_{epoch}.png"), dpi=300)
    plt.close()


def plot_accuracy_curve(acc_history, save_dir):
    epochs = [e for e, _ in acc_history]
    accs = [a for _, a in acc_history]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, accs, marker='o')
    plt.title("Clustering Accuracy vs DEC Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_vs_epoch.png"), dpi=300)
    plt.close()


def create_figure5_panel(snapshot_epochs, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, epoch in enumerate(snapshot_epochs):
        img_path = os.path.join(save_dir, f"tsne_epoch_{epoch}.png")
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            axes[idx].imshow(img)
            axes[idx].axis("off")
            axes[idx].set_title(f"Epoch {epoch}")

    acc_path = os.path.join(save_dir, "accuracy_vs_epoch.png")
    if os.path.exists(acc_path):
        img = plt.imread(acc_path)
        axes[5].imshow(img)
        axes[5].axis("off")
        axes[5].set_title("Accuracy vs Epoch")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figure5_style_panel.png"), dpi=300)
    plt.close()



# ============================================================
# DEC TRAINING WITH FIGURE 5 SNAPSHOTS
# ============================================================
def train_dec(
    dataloader,
    model,
    optimizer_dec,
    tol=config['tol'],
    epochs=20,
    run="epoch",
    snapshot_epochs=(0, 3, 6, 9, 12),
    max_tsne_samples=2000,
    save_dir="dec_fig5_outputs"
):
    prev_assignments = None
    acc_history = []

    # Freeze decoder
    for param in model.decoder.parameters():
        param.requires_grad = False

    epoch = 0

    if run == "tol":
        print(f"\nRunning until only {tol*100}% have changed")
    elif run == "epoch":
        print(f"\nRunning for {epochs} epochs")

    # Snapshot at epoch 0
    if 0 in snapshot_epochs:
        latent_0, labels_0 = get_latent_subset(dataloader, model, max_samples=max_tsne_samples)
        plot_tsne_epoch(latent_0, labels_0, epoch=0, save_dir=save_dir)

    acc0, _, _, _ = evaluate_dec_clustering(dataloader, model)
    acc_history.append((0, acc0))
    print(f"Epoch 0 | Accuracy: {acc0*100:.2f}%")

    while True:
        if run == "epoch" and epoch >= epochs:
            break

        model.eval()
        all_q = []
        with torch.no_grad():
            for X, _ in dataloader:
                X = X.to(device)
                z = model.encode(X)
                all_q.append(model.soft_assign(z).cpu())

        all_q = torch.cat(all_q)
        p_full = target_distribution(all_q)

        current_assignments = all_q.argmax(dim=1).numpy()
        if prev_assignments is not None:
            changed = (current_assignments != prev_assignments).mean()
            print(f"  Epoch {epoch+1}: {changed*100:.2f}% changed")
            if changed < tol:
                print(f"Converged at epoch {epoch+1}")
                break

        prev_assignments = current_assignments.copy()

        p_loader = DataLoader(
            torch.utils.data.TensorDataset(p_full),
            batch_size=dataloader.batch_size,
            shuffle=False
        )

        model.train()
        total_loss = 0.0

        for (X, _), (p_batch,) in zip(dataloader, p_loader):
            X = X.to(device)
            p_batch = p_batch.to(device)

            z = model.encode(X)
            q_batch = model.soft_assign(z)

            loss = F.kl_div((q_batch + 1e-10).log(), p_batch, reduction='batchmean')

            optimizer_dec.zero_grad()
            loss.backward()
            optimizer_dec.step()

            total_loss += loss.item()

        epoch += 1
        print(f"Epoch {epoch:>3d} | KL Loss: {total_loss / len(dataloader):.6f}")

        acc_epoch, _, _, _ = evaluate_dec_clustering(dataloader, model)
        acc_history.append((epoch, acc_epoch))
        print(f"Epoch {epoch:>3d} | Accuracy: {acc_epoch*100:.2f}%")

        if epoch in snapshot_epochs:
            latent_epoch, labels_epoch = get_latent_subset(dataloader, model, max_samples=max_tsne_samples)
            plot_tsne_epoch(latent_epoch, labels_epoch, epoch=epoch, save_dir=save_dir)

    plot_accuracy_curve(acc_history, save_dir)
    create_figure5_panel(snapshot_epochs, save_dir)

    return acc_history


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(torch.__version__)
    print(f"Using {device} device")

    # Dataset
    training_data = datasets.MNIST(
        root="dec/data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.MNIST(
        root="dec/data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Dataloaders
    train_dataloader = DataLoader(training_data, batch_size=config['batch_size'], shuffle=True)
    train_dataloader_noshuffle = DataLoader(training_data, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    # Model
    model = StackedAutoEncoder().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ========================================================
    # Phase 1: Train autoencoder
    # ========================================================
    print("=== Phase 1: Training Autoencoder ===")
    train_autoencoder(train_dataloader, model, loss_fn, optimizer, epochs=config["epochs"])

    # ========================================================
    # Phase 2: K-Means in latent space
    # ========================================================
    print("\n=== Phase 2: K-Means Clustering in Latent Space ===")
    z_train, labels_train = get_latent_vectors(train_dataloader_noshuffle, model)
    z_test, labels_test = get_latent_vectors(test_dataloader, model)

    print(f"Latent vectors shape: {z_train.shape}")

    cluster_assignments_train, kmeans = run_kmeans(z_train, n_clusters=config["n_clusters"])
    cluster_assignments_test = kmeans.predict(z_test)

    print("\n=== Phase 2b: Initial Accuracy ===")
    acc_train, label_mapping = hungarian_accuracy(labels_train, cluster_assignments_train)
    acc_test, _ = hungarian_accuracy(labels_test, cluster_assignments_test)

    print(f"Train clustering accuracy: {acc_train*100:.1f}%")
    print(f"Test clustering accuracy: {acc_test*100:.1f}%")
    print(f"Cluster → Digit mapping: {label_mapping}")

    # ========================================================
    # Phase 3: DEC refinement
    # ========================================================
    print("\n=== Phase 3: Cluster Refinement ===")
    model.init_centroids(kmeans.cluster_centers_)

    optimizer_dec = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"]
    )

    acc_history = train_dec(
        train_dataloader_noshuffle,
        model,
        optimizer_dec,
        epochs=config["refine_epochs"],
        run="epoch",
        snapshot_epochs=config["snapshot_epochs"],
        max_tsne_samples=config["tsne_samples"],
        save_dir=config["figure_dir"]
    )

    # ========================================================
    # Final evaluation
    # ========================================================
    print("\n=== Final Evaluation ===")
    model.eval()
    all_q, all_labels = [], []

    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device)
            z = model.encode(X)
            q = model.soft_assign(z)
            all_q.append(q.cpu())
            all_labels.append(y)

    all_q = torch.cat(all_q)
    all_labels = torch.cat(all_labels).numpy()
    final_assignments = all_q.argmax(dim=1).numpy()

    test_acc, label_mapping = hungarian_accuracy(all_labels, final_assignments)
    print(f"Test clustering accuracy: {test_acc*100:.1f}%")
    print(f"Cluster -> Digit mapping: {label_mapping}")

    print(f"\nSaved outputs in folder: {config['figure_dir']}")
    print("Generated files should include:")
    print("- tsne_epoch_0.png")
    print("- tsne_epoch_3.png")
    print("- tsne_epoch_6.png")
    print("- tsne_epoch_9.png")
    print("- tsne_epoch_12.png")
    print("- accuracy_vs_epoch.png")
    print("- figure5_style_panel.png")