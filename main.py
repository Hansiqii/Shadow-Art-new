import torch
import matplotlib.pyplot as plt
import os
import train_model_functions
import argparse

from train import train
from torch.utils.data import DataLoader
from occupancy_network import OccupancyNetwork
from network_checking import save_loss_plots
from config_manager import create_optimizer, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShadowArt 训练入口")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径（相对或绝对路径皆可）",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(current_dir, config_path)

    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config.get("model", {})
    model = OccupancyNetwork(num_encoding_functions=model_cfg.get("num_encoding_functions", 6))
    model.to(device)

    optimizer = create_optimizer(model, config.get("optimizer", {}))

    torch.save(
        {
            "model.state_dict": model.state_dict(),
            "optimizer.state_dict": optimizer.state_dict(),
        },
        os.path.join(current_dir, "New.pth"),
    )

    loss, draw_loss, real_figure_loss = train(
        current_dir=current_dir,
        device=device,
        model=model,
        optimizer=optimizer,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        config=config,
    )

    save_loss_plots(draw_loss, real_figure_loss)


if __name__ == "__main__":
    main()
