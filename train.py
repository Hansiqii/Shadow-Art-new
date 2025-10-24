import itertools
import os
from typing import Dict, Tuple

import cv2
import torch
from torch.utils.data import DataLoader

import train_model_functions
from RaySamplingDataset import RaySamplingDataset
from config_manager import resolve_path
from diff import compute_diff
from painter import painting
from registration import registration


def train(
    current_dir: str,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    config: Dict,
) -> Tuple[list, list, list]:
    """
    主训练循环，根据配置文件中的参数完成训练。
    """
    training_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})

    input_dir = resolve_path(current_dir, paths_cfg.get("input_dir", "data/input"))
    figures_dir = resolve_path(current_dir, paths_cfg.get("figures_dir", "data/Figures"))
    outcomes_dir = resolve_path(current_dir, paths_cfg.get("outcomes_dir", "outcomes"))
    temp_dir = resolve_path(current_dir, paths_cfg.get("temp_dir", "data/temp"))

    os.makedirs(outcomes_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    image_size = tuple(data_cfg.get("process_image_size", [100, 100]))
    process_images(input_dir, size=image_size)
    process_images(figures_dir, size=image_size)

    epochs = training_cfg.get("epochs", 1)
    batches = training_cfg.get("max_batches_per_epoch", 0)
    batch_size = training_cfg.get("batch_size", 32)
    ray_samples = data_cfg.get("ray_samples_per_pixel", 100)
    eikonal_threshold_ratio = training_cfg.get("eikonal_threshold_ratio", 0.4)

    betas_cfg = training_cfg.get("betas", {})
    base_beta_1 = betas_cfg.get("reg1", 0.0)
    base_beta_2 = betas_cfg.get("reg2", 0.0)
    base_beta_3 = betas_cfg.get("reg3", 0.0)
    base_beta_4 = betas_cfg.get("reg4", 0.0)
    base_beta_5 = betas_cfg.get("reg5", 0.0)
    base_beta_6 = betas_cfg.get("reg6", 0.0)
    base_beta_eikonal = betas_cfg.get("eikonal", 0.0)

    angle_log_cfg = training_cfg.get("angle_log", {})
    log_angles = angle_log_cfg.get("enable", True)
    angles_filename = angle_log_cfg.get("angles_filename", "A_angles.txt")
    screens_filename = angle_log_cfg.get("screens_filename", "A_screens.txt")

    all_losses = []
    draw_losses = []
    real_figure_losses = []

    for epoch in range(epochs):
        angles, screens = generate_angle_screen(model)

        if log_angles:
            save_angles_to_txt(
                angles,
                folder_path=outcomes_dir,
                filename=angles_filename,
            )
            save_angles_to_txt(
                screens,
                folder_path=outcomes_dir,
                filename=screens_filename,
            )

        dataset = RaySamplingDataset(device, figures_dir, angles, screens, n=ray_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        ratio = dataset.area_correction(figures_dir)

        beta_1 = base_beta_1 * (2 ** min(epoch, 3))
        beta_2 = base_beta_2 * (2 ** min(epoch, 3))
        beta_3 = base_beta_3 * (1 if epoch > 3 else 0)
        beta_4 = base_beta_4 * (2 ** max(epoch - 20, 0))
        beta_5 = base_beta_5 * (2 ** epoch)
        beta_6 = base_beta_6 * (1 if epoch > 3 else 0)
        beta_eikonal = base_beta_eikonal * (1 + 0.1 * max(epoch - 20, 0))

        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_losses = train_loop(
            dataset=dataset,
            dataloader=dataloader,
            ratio=ratio,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            batches=batches,
            beta_1=beta_1,
            beta_2=beta_2,
            beta_3=beta_3,
            beta_4=beta_4,
            beta_5=beta_5,
            beta_6=beta_6,
            beta_eikonal=beta_eikonal,
            batch_size=batch_size,
            eikonal_threshold_ratio=eikonal_threshold_ratio,
        )

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
        print(
            f"End of epoch {epoch + 1}, average regular curvature loss: {avg_regular_loss_6:>7f}"
        )
        print(f"End of epoch {epoch + 1}, average eikonal loss: {avg_eikonal_loss:>7f}")

        checkpoint_path = os.path.join(outcomes_dir, f"outcome{epoch + 1}.pth")
        torch.save(
            {
                "model.state_dict": model.state_dict(),
                "optimizer.state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )

        real_figure_loss = painting(model, dataset, current_dir, epoch)
        real_figure_losses.append(real_figure_loss)

        for img_idx in range(len(dataset.images_and_angles)):
            input_image_path = os.path.join(input_dir, f"{img_idx}.png")
            figure_image_path = os.path.join(figures_dir, f"{img_idx}.png")
            outcome_image_path = os.path.join(
                outcomes_dir, f"outcome_Figure{img_idx}_Epoch{epoch + 1}.png"
            )

            compute_diff(
                current_dir,
                input_image_path,
                outcome_image_path,
                epoch,
                img_idx,
                0,
            )
            compute_diff(
                current_dir,
                figure_image_path,
                outcome_image_path,
                epoch,
                img_idx,
                2,
            )

        if epoch % 5 == 4 and epoch > 5:
            print("registration!")
            for img_idx in range(len(dataset.images_and_angles)):
                figure_image_path = os.path.join(figures_dir, f"{img_idx}.png")
                temp_outcome_path = os.path.join(temp_dir, f"outcome_{img_idx}.png")

                registration(
                    current_dir,
                    figure_image_path,
                    temp_outcome_path,
                    epoch,
                    img_idx,
                )
                registration_image_path = os.path.join(
                    outcomes_dir,
                    f"Registration_Figure{img_idx}_Epoch{epoch + 1}.png",
                )
                compute_diff(
                    current_dir,
                    input_image_path,
                    registration_image_path,
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
    batch_size=32,
    eikonal_threshold_ratio=0.4,
):
    """单 epoch 内的训练逻辑。"""
    values = [-1 / dataset.width, 0, 1 / dataset.width]
    A = torch.tensor(list(itertools.product(values, repeat=3)), device=dataset.device)
    losses = []

    size = len(dataloader.dataset)
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    with torch.no_grad():
        test_points = torch.rand(100, 3).to(device) - 0.5
        occupancy = model(test_points)

        print("\n" + "=" * 60)
        print("Model Output Statistics:")
        print(
            f"Occupancy: min={occupancy.min():.3f}, max={occupancy.max():.3f}, mean={occupancy.mean():.3f}"
        )
        print(f"Num near 0 (<0.1): {(occupancy < 0.1).sum()}")
        print(f"Num near 1 (>0.9): {(occupancy > 0.9).sum()}")
        print(f"Num in middle (0.1-0.9): {((occupancy >= 0.1) & (occupancy <= 0.9)).sum()}")
        print("=" * 60 + "\n")

    model.train()

    for batch, (rays, occupancy_values, volumes, img_idxs, rs, cs) in enumerate(
        dataloader
    ):
        if batches != 0 and batch >= batches:
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

        occupancy_estimation = torch.zeros(len(rays), dtype=torch.float32, device=device)

        regularisation_loss_1 = 0
        regularisation_loss_2 = 0
        regularisation_loss_3 = 0
        regularisation_loss_5 = 0
        regularisation_loss_6 = 0
        regularization_loss_term_eikonal_surface_aware = 0

        for ray_id, ray in enumerate(rays):
            if ray.size(0) == 0:
                continue

            occupancy_values_on_whole_ray = model(ray)
            truncate = train_model_functions.truncate_ray(ray)
            occupancy_values_on_ray = occupancy_values_on_whole_ray[truncate[1]]
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
            regularisation_loss_6 += train_model_functions.regularization_loss_term_6(
                dataset, model, ray, img_idx, r, c, A
            )
            regularization_loss_term_eikonal_surface_aware += (
                train_model_functions.regularization_loss_term_eikonal_surface_aware(
                    dataset,
                    model,
                    ray,
                    threshold_ratio=eikonal_threshold_ratio,
                )
            )

        regularisation_loss_4 = 0

        regularisation_loss_1 /= batch_size
        regularisation_loss_2 /= batch_size
        regularisation_loss_3 /= batch_size
        regularisation_loss_5 /= batch_size
        regularisation_loss_6 /= batch_size
        regularization_loss_term_eikonal_surface_aware /= batch_size

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
        optimizer.step()

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
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, filename)

    with open(file_path, "a", encoding="utf-8") as f:
        for angle in angles:
            angle_str = " ".join(str(component.item()) for component in angle.flatten())
            f.write(angle_str + "\n")


def generate_angle_screen(model):
    angles = torch.empty(2, 3)
    angles[0] = torch.tensor([1.0, 0.0, 0.0])
    angles[1] = torch.tensor([0.0, 1.0, 0.0])

    screens = torch.empty(2, 3)
    screens[0] = torch.tensor([1.0, 0.0, 0.0])
    screens[1] = torch.tensor([0.0, 1.0, 0.0])
    return angles, screens


def process_images(folder_path, size=(100, 100)):
    if not os.path.isdir(folder_path):
        print(f"Warning: directory {folder_path} does not exist, skip processing.")
        return

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)

        if not (
            filename.endswith(".jpg")
            or filename.endswith(".png")
            or filename.endswith(".jpeg")
        ):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        resized_img = cv2.resize(binary_img, size, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(img_path, resized_img)
        print(f"Processed and replaced: {img_path}")