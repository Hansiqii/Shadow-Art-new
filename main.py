import torch
import matplotlib.pyplot as plt
from train import train_loop
from train import train
from torch.utils.data import DataLoader
from occupancy_network import OccupancyNetwork
import os
import train_model_functions
import time
from network_checking import save_loss_plots

model = OccupancyNetwork()

learning_rate = 1e-4
batch_size = 32

current_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss function is complicated, will be defined in train loop.

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

torch.save(
    {
        "model.state_dict": model.state_dict(),
        "optimizer.state_dict": optimizer.state_dict(),
    },
    "New.pth",
)

loss, draw_loss, real_figure_loss = train(
    current_dir,
    device,
    model,
    optimizer,
    torch.nn.MSELoss(reduction="mean"),
    epochs=30,
    batches=300000,
    beta_1=0.00001,
    beta_2=0.05,
    beta_3=1e-4,
    beta_4=1e-4,
    beta_5=1e-3,
    beta_6=0,
    beta_eikonal=0,
)

save_loss_plots(draw_loss, real_figure_loss)
