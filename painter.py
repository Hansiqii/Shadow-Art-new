import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.ndimage as ndi
from torch.utils.data import DataLoader
import os
from PIL import Image
from occupancy_network import OccupancyNetwork
from RaySamplingDataset import RaySamplingDataset


def project_network2d_row(img_idx, row, width, model, dataset, reduce=torch.max):
    """处理一行的像素坐标，使用 occupancy 网络进行批量计算."""
    device = next(model.parameters()).device

    # 生成该行所有列的 x 坐标（从 0 到 width 的比例坐标）
    x_coords = torch.linspace(0, 1, steps=width, device=device)

    # 对应的行坐标是固定的，即当前行的 y 坐标（作为比例）
    y_coord = torch.tensor(row / (dataset.height - 1), device=device).repeat(width)

    # 计算出该行所有像素的 (x, y) 坐标
    c = (x_coords * (dataset.width - 1)).long()
    r = (y_coord * (dataset.height - 1)).long()

    # 获取光线 ray，对于每个像素位置 (r, c)
    indices = img_idx * dataset.width * dataset.height + dataset.width * r + c
    rays = torch.stack([dataset[idx][0] for idx in indices]).to(device)
    real_occupancy = torch.stack([dataset[idx][1] for idx in indices]).to(device).unsqueeze(1)

    with torch.no_grad():
        occupancy_values_on_ray = model(rays)
        # estimate_occupancy = result = 1 - torch.prod(1 - occupancy_values_on_ray, dim=1)
        cumprod_result = torch.cumprod(1-occupancy_values_on_ray, dim=0)
        if cumprod_result.size(0) == 0:
            estimate_occupancy = 0
        else:
            estimate_occupancy = 1-cumprod_result[-1]
        # print(real_occupancy)
        # print(real_occupancy.shape)
        # print(estimate_occupancy)
        # print(estimate_occupancy.shape)
        estimate_occupancy = torch.cat([estimate_occupancy, torch.zeros(1, 1, device=estimate_occupancy.device)])
        average_squared_difference = torch.mean((real_occupancy - estimate_occupancy) ** 2)

    # Reduce 光线值，计算该行的结果
    if occupancy_values_on_ray.shape[0] != 0:
        estimated_pixel_values = reduce(occupancy_values_on_ray, dim=1).values
        binary_pixel_values = (estimated_pixel_values > 0.1).float()
    else:
        binary_pixel_values = torch.zeros(width, device=device)

    return binary_pixel_values.squeeze(), average_squared_difference


def painting(model, dataset, current_dir, epoch):
    print("Painting!")
    width, height = dataset.width, dataset.height
    plot_grid = torch.zeros(size=(height, width), device=next(model.parameters()).device)

    total_difference = 0
    for img_idx in range(len(dataset.images_and_angles)):
        # 逐行计算 plot_grid
        for r in range(height):
            plot_grid[r, :], difference = project_network2d_row(img_idx, r, width, model, dataset, torch.max)
            total_difference += difference

        # 将结果转换为 numpy 格式并进行二值化处理
        binary_image = plot_grid.cpu().numpy()
        label_matrix, num_labels = ndi.label(binary_image)
        sizes = np.bincount(label_matrix.ravel())
        sizes[0] = 0  # 忽略背景
        size_threshold = 50
        large_components_mask = np.isin(label_matrix, np.where(sizes > size_threshold))
        filtered_binary_image = large_components_mask & binary_image.astype(bool)
        filtered_binary_image = ((1 - filtered_binary_image) * 255).astype(np.uint8)

        # 保存图像
        output_dir = os.path.join(current_dir, 'outcomes')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        Image.fromarray(filtered_binary_image).save(
            os.path.join(output_dir, f"outcome_Figure{img_idx}_Epoch{epoch + 1}.png"))
        if epoch % 5 == 4 and epoch > 5:
            output_dir = os.path.join(current_dir, 'data/temp')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            Image.fromarray(filtered_binary_image).save(os.path.join(current_dir, f"data/temp/outcome_{img_idx}.png"))
    print("Painting Finish!")
    real_figure_loss = total_difference / height / len(dataset.images_and_angles)
    return real_figure_loss.cpu().numpy()


# current_dir = os.path.dirname(os.path.abspath(__file__))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = OccupancyNetwork()
# checkpoint = torch.load("D:/USTC_CC/ShadowArt with Occupency Network/ShadowArt_Occupency_Network/outcomes/outcome2.pth", map_location=device)
# model.load_state_dict(checkpoint["model.state_dict"])
# data_dir = os.path.join(current_dir, "data/Figures")
# angles_file = os.path.join(current_dir, "data/angle.txt")
# dataset = RaySamplingDataset(device, data_dir, angles_file)
# real_figure_loss = painting(model, dataset, current_dir, 114)