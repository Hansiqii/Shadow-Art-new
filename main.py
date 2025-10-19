import torch
import matplotlib.pyplot as plt
from train import train_loop
from train import train
from torch.utils.data import DataLoader
from occupancy_network import OccupancyNetwork
import os
import train_model_functions
from network_checking import save_loss_plots

model = OccupancyNetwork()

learning_rate = 1e-4
batch_size = 32

current_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss function is complicated, will be defined in train loop.

model.to(device)

# 试图多卡结果不好
# if torch.cuda.device_count() > 1:
#    print(f"Using {torch.cuda.device_count()} GPUs! (DataParallel)")
#    model = torch.nn.DataParallel(model)

# optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=10, line_search_fn= "strong_wolfe")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam([
#     {'params': [model.delta_y_1, model.delta_x_2, model.delta_z_1, model.delta_z_2, model.delta_x_3, model.delta_y_3,
#                 model.screen_y_1, model.screen_x_2, model.screen_z_1, model.screen_z_2, model.screen_x_3, model.screen_y_3],
#      'lr': 5e-4},
#     {'params': [param for name, param in model.named_parameters() if name not in [
#         'delta_y_1', 'delta_x_2', 'delta_z_1', 'delta_z_2', 'delta_x_3', 'delta_y_3',
#         'screen_y_1', 'screen_x_2', 'screen_z_1', 'screen_z_2', 'screen_x_3', 'screen_y_3']],
#      'lr': 1e-4}
# ])
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
    beta_1=0.001,
    beta_2=0.05,
    beta_3=1e-4,
    beta_4=1e-5,
    beta_5=0,
    beta_6=1e-4,
)

save_loss_plots(draw_loss, real_figure_loss)
