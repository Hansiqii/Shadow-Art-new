import torch
from scipy.ndimage import label
import numpy as np
import torch.nn.functional as F

def truncate_ray(ray):
    """
    Params:
        ray: torch.tensor object with shape (n, 3)

    Returns:
        A truncated ray, with rows satisfying the specified conditions:
        - ray[2] is in [0.5, -0.5] (with a small tolerance to avoid precision issues)
        - ray[0] and ray[1] satisfy one of the following:
          1. ray[0] in [0, 0.5] and ray[1] in [0, 0.5] (with a small tolerance)
          2. ray[0] in [0, -0.5] and ray[1] in [0, -0.5] (with a small tolerance)
          3. ray[0] >= 0, ray[1] <= 0, and ray[0] - ray[1] <= 0.5 (with a small tolerance)
          4. ray[0] <= 0, ray[1] >= 0, and ray[0] - ray[1] >= -0.5 (with a small tolerance)
    """
    # Define a small tolerance
    tolerance = 1e-6

    # Condition 1: ray[2] is in [0.5, -0.5] with tolerance
    cond1 = (ray[:, 2] <= 0.5 + tolerance) & (ray[:, 2] >= -0.5 - tolerance)

    # Condition 2: ray[0] and ray[1] satisfy one of the sub-conditions with tolerance
    # cond2_1 = (ray[:, 0] >= 0 - tolerance) & (ray[:, 0] <= 0.5 + tolerance) & (ray[:, 1] >= 0 - tolerance) & (ray[:, 1] <= 0.5 + tolerance)
    # cond2_2 = (ray[:, 0] >= -0.5 - tolerance) & (ray[:, 0] <= 0 + tolerance) & (ray[:, 1] >= -0.5 - tolerance) & (ray[:, 1] <= 0 + tolerance)
    # cond2_3 = (ray[:, 0] >= 0 - tolerance) & (ray[:, 1] <= 0 + tolerance) & ((ray[:, 0] - ray[:, 1]) <= 0.5 + tolerance)
    # cond2_4 = (ray[:, 0] <= 0 + tolerance) & (ray[:, 1] >= 0 - tolerance) & ((ray[:, 0] - ray[:, 1]) >= -0.5 - tolerance)

    # cond2 = cond2_1 | cond2_2 | cond2_3 | cond2_4

    # Combine conditions
    valid_indexes = cond1

    # Filter ray based on the conditions
    return ray[valid_indexes], valid_indexes


def accumulated_occupancy(occupancy_values_on_ray):
    """Params:  occupancy_values_on_ray:    an array of occupancy values,
                                            it should be truncated in the bounding volume.
    
    Returns:    the `probability' of a ray-surface intersection, given the current parameters."""
    # occupancy_values_on_ray = occupancy_values_on_ray.squeeze()
    cumprod_result = torch.cumprod(1-occupancy_values_on_ray, dim=0)
    if cumprod_result.size(0) == 0:
        return 0
    else:
        return 1-cumprod_result[-1]

def regularization_loss_term_1(occupancy_values_on_ray):
    """smooth
    Params:  occupancy_values_on_ray  An array of occupancy values, torch.tensor with dtype torch.float32
                                        and shape (n,). It should be truncated in the bounding volume.
    Returns:    One regularization loss term, the one for this ray."""
    return torch.mean((occupancy_values_on_ray[1:] - occupancy_values_on_ray[:-1])**2)



def regularization_loss_term_2(occupancy_values_on_ray):
    """二值化
    Params:  occupancy_values_on_ray  An array of occupancy values, torch.tensor with dtype torch.float32
                                        and shape (n,). It should be truncated in the bounding volume.
    Returns:    One regularization loss term, the one for this ray."""
    return torch.mean(torch.min(occupancy_values_on_ray**2, (1-occupancy_values_on_ray)**2))

def regularization_loss_term_3(occupancy_values_on_ray, volume, threshold, temperature):
    """总体积
    Params:  occupancy_values_on_ray  An array of occupancy values, torch.tensor with dtype torch.float32
                                        and shape (n,). It should be truncated in the bounding volume.
    Returns:    One regularization loss term, the one for this ray."""
    soft_indicators = torch.sigmoid((occupancy_values_on_ray - threshold) / temperature)
    return torch.sum(soft_indicators * volume)

def regularization_loss_term_5(dataset, model, accumulated_occupancy_, img_idx, r, c):
    """防止薄片
    Params:  occupancy_values_on_ray  An array of occupancy values, torch.tensor with dtype torch.float32
                                        and shape (n,). It should be truncated in the bounding volume.
    Returns:    One regularization loss term, the one for this ray."""
    r_1 = min(dataset.height-1, r+1)
    c_1 = min(dataset.width-1, c+1)
    idx_1 = img_idx * dataset.height * dataset.width + r_1 * dataset.width + c
    idx_2 = img_idx * dataset.height * dataset.width + r * dataset.width + c_1
    f_1 = accumulated_occupancy(model(dataset[idx_1][0]))
    f_2 = accumulated_occupancy(model(dataset[idx_2][0]))
    return (f_1-accumulated_occupancy_)**2 + (f_2-accumulated_occupancy_)**2

def regularization_loss_term_6(dataset, model, ray, img_idx, r, c, A):
    '''梯度一致性正则化'''
    ray = ray.requires_grad_()
    output = model(ray)
    #input_grad = torch.autograd.grad(outputs=output, inputs=ray, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    #norms = torch.norm(input_grad, p=2, dim=1)
    input_grad = compute_norms(model, ray, A)
    norms = torch.norm(input_grad, p=2, dim=1)
    # mean_norms, std_norms = torch.mean(norms), torch.std(norms)
    # norms = torch.where(norms < (2 * std_norms + mean_norms), torch.tensor(0.0), norms)
    # weights = 1 - torch.exp(-0.5 / 5**2 * (norms) ** 2)
    # non_zero_indices = torch.nonzero(weights > 0).squeeze(1)
    # non_zero_weights = weights[non_zero_indices]
    # selected_centers = ray[non_zero_indices]
    # selected_norms = F.normalize(input_grad[non_zero_indices], p=2, dim=1)
    # vec_1s, vec_2s = compute_orthogonal_vectors(selected_norms)
    # circle_points = sample_circle_points(selected_centers, vec_1s, vec_2s, 0.005, 10)
    # circle_points_values = model(circle_points)
    # output_grad = torch.autograd.grad(outputs=circle_points_values, inputs=circle_points,grad_outputs=torch.ones_like(circle_points_values), create_graph=True)[0]
    # output_grad_norms = torch.norm(output_grad, p=2, dim=-1, keepdim=True)
    # output_grad = torch.where(output_grad_norms > 0, output_grad / output_grad_norms, output_grad)
    # selected_norms = selected_norms.unsqueeze(1)
    #loss = torch.sum(non_zero_weights * torch.mean(torch.norm(output_grad - selected_norms, p=1, dim=2), dim=1))

    # another method
    non_zero_indices = torch.nonzero(norms > 0.4 * dataset.width/2).squeeze(1)
    selected_centers = ray[non_zero_indices]
    selected_norms = F.normalize(input_grad[non_zero_indices], p=2, dim=1)
    nearest_points, nearest_distances = find_k_nearest_neighbors(dataset, model, selected_centers, img_idx, r, c, A)
    #nearest_points_values = model(nearest_points)
    #output_grad = torch.autograd.grad(outputs=nearest_points_values, inputs=nearest_points,grad_outputs=torch.ones_like(nearest_points_values), create_graph=True)[0]
    #output_grad_norms = torch.norm(output_grad, p=2, dim=-1, keepdim=True)
    #output_grad = torch.where(output_grad_norms > 0, output_grad / output_grad_norms, output_grad)
    output_grad = compute_norms(model, nearest_points.view(-1,3), A).view(nearest_points.shape)
    output_grad_norms = torch.norm(output_grad, p=2, dim=-1, keepdim=True)
    output_grad = torch.where(output_grad_norms > 0, output_grad / output_grad_norms, output_grad)
    selected_norms = selected_norms.unsqueeze(1)
    loss = torch.mean(torch.mean(torch.div(torch.norm(output_grad - selected_norms, p=1, dim=2), nearest_distances),dim=1))

    if torch.isnan(loss).any():
        return torch.tensor(0.0, device=loss.device)
    else:
        return loss

def regularization_loss_term_eikonal_surface_aware(dataset, model, ray, threshold_ratio=0.4):
    """
    表面感知的 Eikonal 正则化
    仅对接近表面的点（梯度较大的区域）施加约束
    
    Params:
        dataset: 数据集对象
        model: 神经网络模型
        ray: torch.tensor with shape (n, 3)
        threshold_ratio: 用于识别表面点的阈值比例
    
    Returns:
        Eikonal loss (scalar tensor)
    """
    ray = ray.requires_grad_(True)
    occupancy = model(ray)
    
    # 第一次前向传播：计算梯度识别表面点
    gradients = torch.autograd.grad(
        outputs=occupancy,
        inputs=ray,
        grad_outputs=torch.ones_like(occupancy),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradient_norms = torch.norm(gradients, p=2, dim=1)
    
    # 识别表面附近的点（类似于regularization_loss_term_6的逻辑）
    # 这里使用与论文一致的阈值：θ * w
    threshold = threshold_ratio * dataset.width / 2.0
    surface_mask = gradient_norms > threshold
    
    if not surface_mask.any():
        # 如果没有表面点，返回0损失
        return torch.tensor(0.0, device=ray.device)
    
    # 仅对表面点计算 Eikonal loss
    surface_gradient_norms = gradient_norms[surface_mask]
    
    # 目标范数（可以设为1.0或动态计算）
    target_norm = 1.0
    
    eikonal_loss = torch.mean((surface_gradient_norms - target_norm) ** 2)
    return eikonal_loss


# def compute_orthogonal_vectors(normals):
#     device = normals.device
#     batch_size = normals.shape[0]
#
#     vec1 = torch.where(
#         normals[:, 0].unsqueeze(1) != 0,
#         torch.tensor([0.0, 1.0, 0.0], device=device).expand(batch_size, -1),
#         torch.tensor([1.0, 0.0, 0.0], device=device).expand(batch_size, -1)
#     )
#
#     dot_product = torch.sum(normals * vec1, dim=1, keepdim=True)
#     vec1 = vec1 - (dot_product / torch.sum(normals * normals, dim=1, keepdim=True)) * normals
#     vec1 = vec1 / torch.norm(vec1, dim=1, keepdim=True)
#
#     vec2 = torch.cross(normals, vec1, dim=1)
#
#     return vec1, vec2


# def sample_circle_points(centers, vec1, vec2, r, n):
#     angles = torch.linspace(0, 2 * np.pi, n, dtype=torch.float32, device=centers.device)
#     cos_vals = torch.cos(angles)
#     sin_vals = torch.sin(angles)
#     # 扩展维度
#     cos_vals = cos_vals.unsqueeze(0).unsqueeze(2)
#     sin_vals = sin_vals.unsqueeze(0).unsqueeze(2)
#     vec1_expanded = vec1.unsqueeze(1)
#     vec2_expanded = vec2.unsqueeze(1)
#     circle_points = centers.unsqueeze(1) + r * (cos_vals * vec1_expanded + sin_vals * vec2_expanded)
#     return circle_points

def find_k_nearest_neighbors(dataset, model, selected_centers, img_idx, r, c, A):
    databases = find_database(dataset, selected_centers, img_idx, r, c, 1)
    # databases = databases.requires_grad_()
    # output = model(databases)
    # output_grad = torch.autograd.grad(outputs=output, inputs=databases, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    # norms = torch.norm(output_grad, p=2, dim=1)
    # non_zero_indices = torch.nonzero(norms > 5).squeeze(1)
    input_grad = compute_norms(model, databases, A)
    norms = torch.norm(input_grad, p=2, dim=1)
    non_zero_indices = torch.nonzero(norms > 0.4 * dataset.width/2).squeeze(1)
    selected_points = databases[non_zero_indices]
    distances = torch.cdist(selected_centers, selected_points)
    _, nearest_indices = distances.topk(min(6,selected_points.shape[0]), dim=1, largest=False, sorted=False)
    nearest_points = torch.gather(selected_points.expand(len(selected_centers), -1, -1), 1,
                                  nearest_indices.unsqueeze(-1).expand(-1, -1, selected_points.size(-1)))
    nearest_distances = torch.gather(distances, 1, nearest_indices)
    nearest_distances = torch.where(nearest_distances == 0, 1e-8, nearest_distances)
    return nearest_points, nearest_distances

def find_database(dataset, selected_centers, img_idx, r, c, k):
    device = img_idx.device
    rows = torch.arange(max(r - k, 0), min(r + k, dataset.height - 1))
    cols = torch.arange(max(c - k, 0), min(c + k, dataset.width - 1))
    rows_grid, cols_grid = torch.meshgrid(rows, cols, indexing='ij')
    coords = torch.stack([rows_grid.flatten(), cols_grid.flatten()], dim=-1)
    rs = coords[:, 0].to(device)
    cs = coords[:, 1].to(device)
    indices = img_idx * dataset.height * dataset.width + rs * dataset.width + cs
    result = torch.stack([dataset[i][0] for i in indices])
    result = result.reshape(-1,3)
    result_expanded = result.unsqueeze(1)
    selected_centers_expanded = selected_centers.unsqueeze(0)
    matches = torch.all(result_expanded == selected_centers_expanded, dim=2)
    mask = ~torch.any(matches, dim=1)
    result = result[mask]
    return result

def compute_norms(model, points, A):
    cubes = points[:, None, :] + A[None, :, :]
    cubes_outcomes = model(cubes)
    center_outcomes = model(points)
    bs = (cubes_outcomes - center_outcomes[:, None, :]).squeeze(-1)
    A_expanded = A.unsqueeze(0).expand(points.shape[0], -1, -1)
    result = torch.linalg.lstsq(A_expanded, bs).solution
    return result

