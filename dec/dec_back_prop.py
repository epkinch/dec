"""
* TODO implement kmeans
* TODO assign clusters to latent vectors and kmeans space
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
    "tqdm_prints_disable": False,

    # data
    "batch_size": 256,
    "num_workers": 4
}

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# === Defining network ===
class StackedAutoEncoder(nn.Module):
    def __init__(self, *args):
        super().__init__(*args)
        self.n_clusters = params["n_clusters"]
        self.alpha = params["alpha"]
        self.latent_dim = params["latent_dim"]
        self.flatten = nn.Flatten()
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
        return self.encoder(self.flatten(x))

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

net = StackedAutoEncoder()
net.to(device)

# === Generating data and laoders ===
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
dataset = datasets.MNIST("dec/data", train=True, transform=transform, download=True)
train_dataset = Subset(dataset, range(50000))
test_dataset = Subset(dataset, range(50000, len(dataset)))

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=params["batch_size"],
    num_workers=params["num_workers"],
)
test_loader = DataLoader(
    test_dataset, shuffle=False, batch_size=1, num_workers=params["num_workers"],
)

# === Set up optimizer ===
optimizer = torch.optim.SGD(
        net.parameters(),
        lr=params["lr"]
    )

# === Create loss criterion ===
criterion = torch.nn.MSELoss()

# === INITIALIZING THE MODEL ===
def init_net():
    step = 0
    loss_all = list()
    step_all = list()
    train_acc_all = list()

    for epoch in tqdm( range(params["init_epochs"]), disable=params["tqdm_prints_disable"] ):
        net.train()
        for batch_index, (x, y) in tqdm( enumerate(train_loader), disable=False):
            step+=1

            x = x.to(device)
            y = y.to(device)

            z, x_recon = net(x)
            q = net.soft_assign(z)
            cluster = q.argmax(dim=1)

            loss = criterion(x_recon, x.view(x.size(0), -1))
            # reset the gradients
            optimizer.zero_grad(set_to_none=True)
            # backward to compute the gradients
            loss.backward()

            # take a gradient descent step
            optimizer.step()

            correct_indicators = z.max(1)[1].data == y
            train_acc = correct_indicators.sum().item() / y.shape[0]

            if step %400 == 0:
                print("train_acc", train_acc)   

            loss_all.append(loss.item()) # only keeps the loss data and gets rid of gradient
            step_all.append(step)
            train_acc_all.append(train_acc)
    return loss_all, step_all, train_acc_all

if __name__ == "__main__":
    loss_all, step_all, train_acc_all = init_net()
    print("hello world")