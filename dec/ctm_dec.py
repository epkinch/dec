import torch
from torch import nn
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torchvision.transforms import v2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

params = {
    ### model
    "latent_dim": 10,
    "alpha": 1.0,
    ### data
    "dataset": "MNIST",
    "data_dim": 784,
    "n_clusters": 10,
    ### data loader
    "shuffle": True,
    "batch_size": 128,
    "num_workers": 4,
    ### optimizer
    "optimizer_name": "SGD",
    "optimizer_lr": 1e-2,
    "optimizer_weight_decay": 0,
    ### optimizer_scheduler
    "optimizer_scheduler": "StepLR",
    "optimizer_scheduler_gamma": None,
    "optimizer_scheduler_patience": None,
    "optimizer_scheduler_tmax": None,
    "optimizer_scheduler_stepsize": 10,
    "optimizer_scheduler_gamma": 0.5,
    ### loss
    # "loss_name": "CrossEnt",
    "loss_name": "MSE",
    ### train related
    "epochs": 10,
    "tqdm_prints_disable": False,
    "disable_inner_loop": False,
    }

# device = "cpu" # for cpu
device = "cuda:0" # for nvidia

# Download training data from open datasets.
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
dataset = torchvision.datasets.MNIST(root="dec/data", train=True, transform=transform, download=True)
train_dataset = torch.utils.data.Subset(dataset, range(50000))
test_dataset = torch.utils.data.Subset(dataset, range(50000, len(dataset)))

# create loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=params["shuffle"],
    batch_size=params["batch_size"],
    num_workers=params["num_workers"],
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, shuffle=False, batch_size=1, num_workers=params["num_workers"],
)

# Define model
class StackedAutoEncoder(nn.Module):
    def __init__(self, n_clusters=params['n_clusters'], latent_dim=params['latent_dim'], alpha=params['alpha']):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(params["data_dim"], 500), # Input layer to first hidden layer
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 2000), # Latent representation (bottleneck)
            nn.ReLU(True),
            nn.Linear(2000, params["latent_dim"]) # Deepest layer of encoder
        )
        self.decoder = nn.Sequential(
            nn.Linear(params["latent_dim"], 2000),
            nn.ReLU(True),
            nn.Linear(2000, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, params["data_dim"]), # Output layer, same size as input
            nn.Sigmoid() # Use Sigmoid to ensure output pixel values are in [0, 1] range
        )
        # Centroids stored directly on the model — initialized later from K-Means
        self.centroids = nn.Parameter(
            torch.randn(n_clusters, latent_dim),
            requires_grad=False
        )
        
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
        z = self.encoder(self.flatten(x))
        x_recon = self.decoder(z)
        return F.log_softmax(x_recon), z
    
net = StackedAutoEncoder()
net.to(device)


optimizer = torch.optim.SGD(
    net.parameters(), # paramters to optimize
    lr=params["optimizer_lr"],
    weight_decay=params["optimizer_weight_decay"],  # l2 norm regularization
)

if params["optimizer_scheduler"] == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params["optimizer_scheduler_stepsize"],
        gamma=params["optimizer_scheduler_gamma"],
    )

if params["loss_name"] == "MSE":
    criterion = torch.nn.MSELoss()
elif params["loss_name"] == "L1loss":
    criterion = torch.nn.L1Loss()
elif params["loss_name"] == "SmoothL1Loss":
    criterion = torch.nn.SmoothL1Loss()
elif params["loss_name"] == "CrossEnt":
    criterion =  torch.nn.CrossEntropyLoss()

# train loop
def train_init():
    print("start training dnn.")

    net.train()

    step = 0
    loss_all = list()
    step_all = list()
    train_acc_all = list()

    for epoch in tqdm( range(params["epochs"]), disable=params["tqdm_prints_disable"] ):
        net.train()
        
        #per iteration loop
        for i, (x,y) in tqdm(enumerate(train_loader)):
            step += 1
            x = x.to(device)
            y = y.to(device)
            #forward prediction
            x_recon, _ = net(x)
            # calculate loss
            loss = criterion(x_recon, x.view(x.size(0), -1))
            # reset the gradients
            optimizer.zero_grad(set_to_none=True)
            # backward to compute the gradients
            loss.backward()

            # take a gradient descent step
            optimizer.step()

            correct_indicators = x_recon.max(1)[1].data == y.view(y.size(0), -1)
            train_acc = correct_indicators.sum().item() / y.shape[0]
            
            if step % 400 == 0:
                print("train_acc", train_acc)   

        loss_all.append(loss.item()) # only keeps the loss data and gets rid of gradient
        step_all.append(step)
        train_acc_all.append(train_acc)

    # scheduler
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(loss)
    else:
        scheduler.step()

if __name__ == '__main__':
    train_init()