"""
* TODO implement kmeans
* TODO assign clusters to latent vectors in kmeans space
* TODO calculate accuracy
"""
import torch
from torch import nn
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms import v2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

params = {
    # model
    "lr": 1e-2,
    "optimizer": "SGD",
    "loss": "MSE",
    "alpha": 1.0,
    "latent_dim": 10,
    "n_clusters": 10,

    # training
    "init_epochs": 10,
    "refine_epochs": 10,
    "tqdm_prints_disable": True,

    # data
    "batch_size": 256,
    "num_workers": 4
}

# === Defining network ===
class StackedAutoEncoder(nn.Module):
    def __init__(self, *args):
        super().__init__(*args)
        self.n_clusters = params["n_clusters"]
        self.alpha = params["alpha"]
        self.latent_dim = params["latent_dim"]
        # self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 28*28),
            nn.Sigmoid()
        )

        self.centroids = nn.Parameter(
            torch.randn(self.n_clusters, self.latent_dim),
            requires_grad=False
        )

    def encode(self, x):
        # return self.encoder(self.flatten(x))
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decoder(z)
        return z, x_recon

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

# === Generating data and laoders ===
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
dataset = datasets.MNIST("dec/data", train=True, transform=transform, download=True)
train_dataset = Subset(dataset, range(50000))
test_dataset = Subset(dataset, range(50000, len(dataset)))

# === INITIALIZING THE MODEL ===
# def init_net():
#     step = 0
#     loss_all = list()
#     step_all = list()

#     for epoch in tqdm( range(params["init_epochs"]), disable=params["tqdm_prints_disable"] ):
#         net.train()
#         for batch_index, (x, y) in tqdm( enumerate(train_loader), disable=False):
#             step+=1

#             x = torch.flatten(x, 1)

#             x = x.to(device)
#             y = y.to(device)


#             z, x_recon = net(x)

#             # loss = criterion(x_recon, x.view(x.size(0), -1))
#             loss = criterion(x_recon, x)
#             # reset the gradients
#             optimizer.zero_grad(set_to_none=True)
#             # backward to compute the gradients
#             loss.backward()
#             # take a gradient descent step
#             optimizer.step()

#             if step % 100 == 0:
#                 print(f"loss: {loss}")

#             loss_all.append(loss.item()) # only keeps the loss data and gets rid of gradient
#             step_all.append(step)
#     return loss_all, step_all

def init_net(dataloader, model, loss_fn, optimizer, epochs=20):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch, (x, _) in enumerate(dataloader):          # _ = ignore labels entirely during AE training
            x = torch.flatten(x, 1)
            x = x.to(device)
            x_recon, z = model(x)

            # Loss is pixel reconstruction error — shape [batch,784] vs [batch,784]
            loss = loss_fn(x_recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch+1:>3d} | Recon Loss: {avg:.6f}")

# === Perfrom kmeans on latent space
def run_kmeans(z, n_clusters=10, n_init=20, max_iter=300):
    kmeans = KMeans(
        n_clusters=n_clusters,
        init = "k-means++",
        n_init = n_init,
        max_iter = max_iter,
        random_state = 42
    )
    cluster_assignments = kmeans.fit_predict(z)
    return cluster_assignments, kmeans

# === Extract latent vectors from the entire dataset ===
def get_latent_vectors(dataloader, model):
    model.eval()
    all_z, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = torch.flatten(x, 1)
            x = x.to(device)
            _, z = model(x)
            all_z.append(z.cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_z), np.concatenate(all_labels)

# === Compute accuracy from cost matrix ===
def accuracy(true_labels, cluster_assignments, n_clusters=10, n_classes=10):
    cost_matrix = np.zeros((n_clusters, n_classes))
    for cluster_id, true_id in zip(cluster_assignments, true_labels):
        cost_matrix[cluster_id, true_id] += 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    remapped = np.array([mapping.get(c, -1) for c in cluster_assignments])
    return accuracy_score(true_labels, remapped), mapping

if __name__ == "__main__":

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=params["num_workers"],
    )

    net = StackedAutoEncoder().to(device)

    # === Set up optimizer ===
    optimizer = torch.optim.SGD(
            net.parameters(),
            lr=params["lr"]
        )

    # === Create loss criterion ===
    criterion = torch.nn.MSELoss()

    # loss_all, step_all = init_net()
    init_net(train_loader, net, criterion, optimizer, params['init_epochs'])
    z_train, labels_train = get_latent_vectors(train_loader, net)
    z_test, labesl_test = get_latent_vectors(test_loader, net)
    clusters_train, kmeans = run_kmeans(z_train)
    clusters_test = kmeans.fit_predict(z_test)

    acc_train, mapping = accuracy(labels_train, clusters_train)
    acc_test, _ = accuracy(labesl_test, clusters_test)
    print(f"Train clustering accuracy: {acc_train*100:.1f}%")
    print(f"Test clustering accuracy: {acc_test*100:.1f}%")
    print(f"Cluster → Digit mapping: {mapping}")