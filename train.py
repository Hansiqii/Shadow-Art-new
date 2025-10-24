import torch
import torch.optim as optim
import numpy as np
import torchviz
import itertools
from occupancy_network import OccupancyNetwork
from RaySamplingDataset import RaySamplingDataset
from torch.utils.data import DataLoader
import train_model_functions
import os
import cv2
from painter import painting
from registration import registration
from diff import compute_diff
from data_preprocessing import load_images_and_angles
from RaySamplingDataset import RaySamplingDataset

# Initialize model
# model = OccupancyNetwork()
#
# # Hyperparameters
# learning_rate = 1e-3
batch_size = 32
#


# Define the training loop
def train(
    current_dir,
    device,
    model,
    optimizer,
    loss_fn,
    epochs,
    batches,
    beta_1,
    beta_2,
    beta_3,
    beta_4,
    beta_5,
    beta_6,
    beta_eikonal,
):
    """Trains the model' over multiple epochs using the train_loop' function.

    Parameters:
            dataloader:     A torch.dataset.DataLoader object that renders items from a dataset. These items
                            should have the form: a 2-tuple of a torch.tensor object of shape (n, 3) and an occupancy value
                            in [0,1].
            model:          The occupancy network. This is a 3D to 1D NN.
            optimizer:      The optimizer to do the steps, it could take e.g. the model.parameters() as parameters.
            epochs:         The number of epochs to train the network.
            batches:        The number of batches to train per epoch, 0 for the whole dataset.
            beta:           The hyperparameter for the loss function.

    Returns:
            all_losses:     A list of the losses for each epoch."""
    all_losses = []
    draw_losses = []
    real_figure_losses = []
    input_dir = os.path.join(current_dir, "data/input")
    data_dir = os.path.join(current_dir, "data/Figures")
    process_images(input_dir)
    process_images(data_dir)
    for epoch in range(epochs):
        data_dir = os.path.join(current_dir, "data/Figures")
        # angles_file = os.path.join(current_dir, "data/angle.txt")
        # process_images_in_directory(data_dir)

        # 动态生成光源与屏幕方向
        angles, screens = generate_angle_screen(model)
        save_angles_to_txt(
            angles, 
            folder_path=os.path.join(current_dir, "outcomes"),
            filename="A_angles.txt",
            )
        save_angles_to_txt(
            screens,
            folder_path=os.path.join(current_dir, "outcomes"),
            filename="A_screens.txt",
        )

        # 构造“Ray-Occupancy 数据集”
        dataset = RaySamplingDataset(device, data_dir, angles, screens)
        # dataset.__getitem__(2500+800+16)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        ratio = dataset.area_correction(data_dir)

        # 核心训练调用
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_losses = train_loop(
            dataset,
            dataloader,
            ratio,
            model,
            optimizer,
            loss_fn,
            batches,
            beta_1 * 2 ** min(epoch, 3),
            beta_2 * 2 ** min(epoch, 3),
            beta_3 * (1 if 3 < epoch else 0),
            beta_4 * 2 ** max(epoch-20, 0),
            beta_5 * 2 ** (epoch),
            beta_6 * (1 if epoch > 3 else 0),
            beta_eikonal * (1 + 0.1 * max(epoch - 20, 0)),
            # beta_eikonal * (1 + 0.1 * max(epoch - 20, 0)),
        )

        # 计算平均损失+打印日志
        all_losses.append(epoch_losses)
        avg_loss = sum(loss[0] for loss in epoch_losses) / len(epoch_losses)
        avg_figure_loss = sum(loss[1] for loss in epoch_losses) / len(epoch_losses)
        avg_regular_loss_1 = sum(loss[2] for loss in epoch_losses) / len(epoch_losses)
        avg_regular_loss_2 = sum(loss[3] for loss in epoch_losses) / len(epoch_losses)
        avg_regular_loss_3 = sum(loss[4] for loss in epoch_losses) / len(epoch_losses)
        avg_regular_loss_4 = sum(loss[5] for loss in epoch_losses) / len(epoch_losses)
        avg_regular_loss_5 = sum(loss[6] for loss in epoch_losses) / len(epoch_losses)
        avg_regular_loss_6 = sum(loss[7] for loss in epoch_losses) / len(epoch_losses)
        avg_eikonal_loss = sum(loss[8] for loss in epoch_losses) / len(epoch_losses)
        draw_losses.append(
            (
                avg_loss,
                avg_figure_loss,
                avg_regular_loss_1,
                avg_regular_loss_2,
                avg_regular_loss_3,
                avg_regular_loss_6,
            )
        )
        print(f"End of epoch {epoch + 1}, average loss: {avg_loss:>7f}")
        print(f"End of epoch {epoch + 1}, average figure loss: {avg_figure_loss:>7f}")
        print(
            f"End of epoch {epoch + 1}, average regular each near loss: {avg_regular_loss_1:>7f}"
        )
        print(
            f"End of epoch {epoch + 1}, average regular near 0/1 loss: {avg_regular_loss_2:>7f}"
        )
        print(
            f"End of epoch {epoch + 1}, average regular minimum volume loss: {avg_regular_loss_3:>7f}"
        )
        # print(f"End of epoch {epoch + 1}, average regular brick loss: {avg_regular_loss_4:>7f}")
        # print(f"End of epoch {epoch + 1}, average regular brick loss: {avg_regular_loss_5:>7f}")
        print(
            f"End of epoch {epoch + 1}, average regular curvature loss: {avg_regular_loss_6:>7f}"
        )
        print(f"End of epoch {epoch + 1}, average eikonal loss: {avg_eikonal_loss:>7f}")
        
        output_dir = os.path.join(current_dir, "outcomes")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存模型与可视化
        torch.save(
            {
                "model.state_dict": model.state_dict(),
                "optimizer.state_dict": optimizer.state_dict(),
            },
            os.path.join(current_dir, f"outcomes/outcome{epoch+1}.pth"),
        )

        real_figure_loss = painting(model, dataset, current_dir, epoch)
        real_figure_losses.append(real_figure_loss)

        # 投影差异计算
        for img_idx in range(len(dataset.images_and_angles)):
            compute_diff(
                current_dir,
                os.path.join(current_dir, f"data/input/{img_idx}.png"),
                os.path.join(
                    current_dir, f"outcomes/outcome_Figure{img_idx}_Epoch{epoch+1}.png"
                ),
                epoch,
                img_idx,
                0,
            )
            compute_diff(
                current_dir,
                os.path.join(current_dir, f"data/Figures/{img_idx}.png"),
                os.path.join(
                    current_dir, f"outcomes/outcome_Figure{img_idx}_Epoch{epoch+1}.png"
                ),
                epoch,
                img_idx,
                2,
            )

        # 每隔5个epoch进行一次registration
        if epoch % 5 == 4 and epoch > 5:
            print("registration!")
            for img_idx in range(len(dataset.images_and_angles)):
                registration(
                    current_dir,
                    os.path.join(current_dir, f"data/Figures/{img_idx}.png"),
                    os.path.join(current_dir, f"data/temp/outcome_{img_idx}.png"),
                    epoch,
                    img_idx,
                )
                compute_diff(
                    current_dir,
                    os.path.join(current_dir, f"data/input/{img_idx}.png"),
                    os.path.join(
                        current_dir,
                        f"outcomes/Registration_Figure{img_idx}_Epoch{epoch+1}.png",
                    ),
                    epoch,
                    img_idx,
                    1,
                )
    return all_losses, draw_losses, real_figure_losses


def train_loop(
    dataset,
    dataloader,
    ratio,
    model,
    optimizer,
    loss_fn,
    batches=0,
    beta_1=0.1,
    beta_2=0.1,
    beta_3=5e-4,
    beta_4=0.1,
    beta_5=0.1,
    beta_6=0.1,
    beta_eikonal=0.1,
):
    """Trains the model, which has to be a 3D to 1D occupancy network on a dataloader which has to provide
    rays of coordinates and corresponding occupancy values as items. The optimizer will be doing steps, loss is defined
    in this function.

    Parameters:
            dataloader:     A torch.dataset.DataLoader object that renders items from a dataset. These items
                            should have the form: a 2-tuple of a torch.tensor object of shape (n, 3) and an occupancy value
                            in [0,1].
            model:          The occupancy network. This is a 3D to 1D NN.
            optimizer:      The optimizer to do the steps, it could take e.g. the model.parameters() as parameters.
            batches:        The number of batches to train the network, 0 for the whole dataset.
            beta:           The hyperparameter for the loss function.

    Returns:
            losses:         A list of the losses.
    """
    # 初始化
    values = [-1 / dataset.width, 0, 1 / dataset.width]
    A = torch.tensor(list(itertools.product(values, repeat=3)), device=dataset.device)
    losses = []

    size = len(dataloader.dataset)
    model.train()  # Set the model to training mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # ========== 添加调试代码 ==========
    model.eval()  # 临时切换到评估模式
    with torch.no_grad():
        # 测试100个随机点
        test_points = torch.rand(100, 3).to(device) - 0.5
        
        # 直接调用模型（不管内部结构）
        occupancy = model(test_points)
        
        print("\n" + "="*60)
        print("Model Output Statistics:")
        print(f"Occupancy: min={occupancy.min():.3f}, max={occupancy.max():.3f}, mean={occupancy.mean():.3f}")
        print(f"Num near 0 (<0.1): {(occupancy < 0.1).sum()}")
        print(f"Num near 1 (>0.9): {(occupancy > 0.9).sum()}")
        print(f"Num in middle (0.1-0.9): {((occupancy >= 0.1) & (occupancy <= 0.9)).sum()}")
        print("="*60 + "\n")
    
    model.train()  # 恢复训练模式
    # ========== 调试代码结束 ==========
    
    for batch, (rays, occupancy_values, volumes, img_idxs, rs, cs) in enumerate(
        dataloader
    ):
        if not batches == 0 and batch >= batches:
            break

        rays, occupancy_values, volumes, img_idxs, rs, cs = (
            rays.to(device),
            occupancy_values.to(device),
            volumes.to(device),
            img_idxs.to(device),
            rs.to(device),
            cs.to(device),
        )
        optimizer.zero_grad()

        occupancy_estimation = torch.zeros(
            len(rays), dtype=torch.float32, device=device
        )

        regularisation_loss_1 = 0
        regularisation_loss_2 = 0
        regularisation_loss_3 = 0
        regularisation_loss_5 = 0
        regularisation_loss_6 = 0
        regularization_loss_term_eikonal_surface_aware = 0

        for ray_id, ray in enumerate(rays):
            if ray.size(0) == 0:
                continue

            # occupancy 估计
            occupancy_values_on_whole_ray = model(ray)
            truncate = train_model_functions.truncate_ray(ray)
            occupancy_values_on_ray = occupancy_values_on_whole_ray[truncate[1]]
            # print(occupancy_values_on_ray)
            accumulated_occupancy = train_model_functions.accumulated_occupancy(
                occupancy_values_on_ray
            )
            occupancy_estimation[ray_id] = accumulated_occupancy

            regularisation_loss_1 += train_model_functions.regularization_loss_term_1(
                occupancy_values_on_whole_ray
            )
            regularisation_loss_2 += train_model_functions.regularization_loss_term_2(
                occupancy_values_on_whole_ray
            )
            volume = volumes[ray_id]
            volume = volume[truncate[1]]
            regularisation_loss_3 += train_model_functions.regularization_loss_term_3(
                occupancy_values_on_ray, volume, threshold=0.1, temperature=1e-3
            )
            img_idx, r, c = img_idxs[ray_id], rs[ray_id], cs[ray_id]
            # regularisation_loss_5 += train_model_functions.regularization_loss_term_5(dataset, model, accumulated_occupancy, img_idx, r, c)
            regularisation_loss_6 += train_model_functions.regularization_loss_term_6(
                dataset, model, ray, img_idx, r, c, A
            )
            regularization_loss_term_eikonal_surface_aware += (
                train_model_functions.regularization_loss_term_eikonal_surface_aware(
                    dataset, model, ray, threshold_ratio=0.4
                )
            )

        regularisation_loss_4 = 0
        # if batch % 100 == 0 and batch > 0:
        #     regularisation_loss_4 = compute_total_integral(model, grid_tensor, 0.01, 100, threshold=1e-1)
        #     regularisation_loss_4 = torch.tensor(regularisation_loss_4, dtype=torch.float32, requires_grad=True)
        #     print(regularisation_loss_4)

        # 归一化
        regularisation_loss_1 /= batch_size
        regularisation_loss_2 /= batch_size
        regularisation_loss_3 /= batch_size
        regularisation_loss_5 /= batch_size
        regularisation_loss_6 /= batch_size
        regularization_loss_term_eikonal_surface_aware /= batch_size
        # 主渲染损失rendering loss
        img_loss = loss_fn(occupancy_estimation, occupancy_values)
        img_loss = img_loss * ratio

        loss = 2 * (
            beta_1 * regularisation_loss_1
            + beta_2 * regularisation_loss_2
            + beta_3 * regularisation_loss_3
            + beta_4 * regularisation_loss_4
            + beta_5 * regularisation_loss_5
            + beta_6 * regularisation_loss_6
            + img_loss
            + beta_eikonal * regularization_loss_term_eikonal_surface_aware
        )
        # loss = img_loss
        losses.append(
            (
                float(loss),
                img_loss.item(),
                float(regularisation_loss_1),
                float(regularisation_loss_2),
                float(regularisation_loss_3),
                float(regularisation_loss_4),
                float(regularisation_loss_5),
                float(regularisation_loss_6),
                float(regularization_loss_term_eikonal_surface_aware),
            )
        )

        loss.backward(retain_graph=True)
        # print(model.delta_x.grad)
        # print(model.delta_y.grad)
        # dot = torchviz.make_dot(loss, params=dict(
        #     list(model.named_parameters()) + [('delta_x', model.delta_x), ('delta_y', model.delta_y)]))
        # dot.render("delta_x_computational_graph", format="png")  # 保存计算图为 PNG 文件
        optimizer.step()

        # print
        if batch % 10 == 0:
            current = batch * len(rays)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
            print("figure:", img_loss)
            print("reg1:", regularisation_loss_1)
            print("reg2:", regularisation_loss_2)
            print("reg3:", regularisation_loss_3)
            print("reg5:", regularisation_loss_5)
            print("reg6:", regularisation_loss_6)
            print("reg7:", regularization_loss_term_eikonal_surface_aware)
    return losses



def compute_integral(model, p, r, num_samples, f_p):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = p.to(device)
    f_p = f_p.to(device)
    samples = torch.randn((num_samples, 3))
    samples = samples / torch.norm(samples, dim=1, keepdim=True)
    samples = samples * torch.rand(num_samples, 1) * r
    samples = samples.to(device)
    samples = samples + p

    f_samples = model(samples)
    integral = torch.mean((f_samples - f_p) ** 2)

    return integral


def compute_total_integral(model, grid, r, num_samples, threshold=0.1):
    total_integral = 0.0
    i = 1
    for point in grid:
        f_p = model(point)

        if f_p > threshold:
            integral_value = compute_integral(model, point, r, num_samples, f_p)
            total_integral += integral_value.item()
            i = +1

    return total_integral / i


def save_angles_to_txt(angles, folder_path="D:/your_path/", filename="A_angles.txt"):
    '''把本轮训练生成的光源方向（或屏幕法向）参数，保存到一个 A_angles.txt文件中'''
    # 如果文件夹不存在，创建该文件夹
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 构建完整的文件路径
    file_path = os.path.join(folder_path, filename)

    # 写入文件
    with open(file_path, "a") as f:
        for angle in angles:
            angle_str = " ".join(map(str, angle))
            f.write(angle_str + "\n")


def generate_angle_screen(model):
    '''生成本轮训练中使用的光照方向（angles）与屏幕法向（screens）向量'''
    angles = torch.empty(2, 3)  # 创建一个空的张量
    angles[0] = torch.tensor([1.0, 0.0, 0.0])  # 第一个行
    # angles[0, 0] = 1.0  # 第二个行，第一列
    # angles[0, 1] = model.delta_y_1  # 第二个行，第二列
    # angles[0, 2] = model.delta_z_1  # 第二个行，第三列
    angles[1] = torch.tensor([0.0, 1.0, 0.0])
    # angles[1, 0] = 1.0 + model.delta_x_2  # 第二个行，第一列
    # angles[1, 1] = 1.0  # 第二个行，第二列
    # angles[1, 2] = 0.0  # 第二个行，第三列
    # angles[2, 0] = model.delta_x_3  # 第二个行，第一列
    # angles[2, 1] = model.delta_y_3  # 第二个行，第二列
    # angles[2, 2] = 1.0 # 第二个行，第三列
    # angles[2] = torch.tensor([0.0, 0.0, 1.0])
    screens = torch.empty(2, 3)  # 创建一个空的张量
    screens[0] = torch.tensor([1.0, 0.0, 0.0])  # 第一个行
    # screens[0, 0] = 1.0  # 第二个行，第一列
    # screens[0, 1] = model.screen_y_1  # 第二个行，第二列
    # screens[0, 2] = model.screen_z_1  # 第二个行，第三列
    screens[1] = torch.tensor([0.0, 1.0, 0.0])
    # screens[1, 0] = model.screen_x_2  # 第二个行，第一列
    # screens[1, 1] = 1.0  # 第二个行，第二列
    # screens[1, 2] = model.screen_z_2  # 第二个行，第三列
    # screens[2, 0] = model.screen_x_3  # 第二个行，第一列
    # screens[2, 1] = model.screen_y_3  # 第二个行，第二列
    # screens[2, 2] = 1.0 # 第二个行，第三列
    # screens[2] = torch.tensor([0.0, 0.0, 1.0])
    return angles, screens


def process_images(folder_path, size=(100, 100)):
    '''批量预处理数据集中所有输入图像，确保每张图都是统一大小的二值影子图'''
    # 遍历文件夹中的每张图片
    for filename in os.listdir(folder_path):
        # 构建完整路径
        img_path = os.path.join(folder_path, filename)

        # 检查是否为图片文件
        if not (
            filename.endswith(".jpg")
            or filename.endswith(".png")
            or filename.endswith(".jpeg")
        ):
            continue

        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 检查图像是否读取成功
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        # 转换为二值图像
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 调整大小为 50x50
        resized_img = cv2.resize(binary_img, size, interpolation=cv2.INTER_NEAREST)

        # 覆盖原始图片
        cv2.imwrite(img_path, resized_img)
        print(f"Processed and replaced: {img_path}")


# process_images("D:/USTC_CC/ShadowArt with Occupency Network/ShadowArt_Occupency_Network/data/Figures")