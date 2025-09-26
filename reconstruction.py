import numpy as np
import torch
import mcubes
from occupancy_network import OccupancyNetwork
import os

# Step 1: Define the grid of points in 3D space
def create_grid(N, xmin, xmax, ymin, ymax, zmin, zmax):
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)
    grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), -1)
    return grid

# Step 2: Evaluate the occupancy network at each point in the grid
def evaluate_occupancy_network(model, grid, device):
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
        grid_tensor = grid_tensor.view(-1, 3)  # Flatten the grid tensor
        occupancy_values = model(grid_tensor)
        occupancy_values = occupancy_values.view(grid.shape[:-1])
    return occupancy_values.cpu().numpy()

# Step 3: Use the Marching Cubes algorithm to extract the iso-surface
def extract_mesh(occupancy_values, grid, threshold=1e-2):
    occupancy_values = occupancy_values > 0.08
    occupancy_values = mcubes.smooth(occupancy_values)
    vertices, triangles = mcubes.marching_cubes(occupancy_values, 0.5)
    
    # Define a small tolerance
    tolerance = 1e-6
    
    cond2_1 = (vertices[:, 0] >= (0.5 - tolerance)*200) & (vertices[:, 0] <= (1 + tolerance)*200) & (vertices[:, 1] >= (0.5 - tolerance)*200) & (vertices[:, 1] <= (1 + tolerance)*200)
    cond2_2 = (vertices[:, 0] >=  - tolerance*200) & (vertices[:, 0] <= (0.5 + tolerance)*200) & (vertices[:, 1] >= -tolerance*200) & (vertices[:, 1] <= (0.5 + tolerance)*200)
    cond2_3 = (vertices[:, 0] >= (0.5 - tolerance)*200) & (vertices[:, 1] <= (0.5 + tolerance)*200) & ((vertices[:, 0] - vertices[:, 1]) <= (0.5 + tolerance)*200)
    cond2_4 = (vertices[:, 0] <= (0.5 + tolerance)*200) & (vertices[:, 1] >= (0.5 - tolerance)*200) & ((vertices[:, 0] - vertices[:, 1]) >= (-0.5- tolerance)*200)
    cond2 = cond2_1 | cond2_2 | cond2_3 | cond2_4
    # Filter vertices based on conditions
    # valid_mask = (
    #     (vertices[:, 2] <= (1 + tolerance)*200) & (vertices[:, 2] >=  - tolerance*200) &
    #     cond2
    # )
    valid_mask = (
        (vertices[:, 2] <= (1 + tolerance)*200) & (vertices[:, 2] >=  - tolerance*200)
    )
    
    # Apply mask to filter valid vertices
    valid_vertices = vertices[valid_mask]

    # 构建一个新的索引映射，-1 表示无效
    vertex_mapping = -np.ones(len(vertices), dtype=int)
    vertex_mapping[valid_mask] = np.arange(len(valid_vertices))

    # 重新构建三角形
    valid_triangles = []
    for t in triangles:
        mapped_triangle = vertex_mapping[t]
        if np.all(mapped_triangle >= 0):  # 只有所有顶点都有效时才保留
            valid_triangles.append(mapped_triangle)
    
    valid_triangles = np.array(valid_triangles, dtype=int)  # 确保是整数数组
    
    # return vertices, triangles
    return valid_vertices/200-0.5, valid_triangles

# Step 4: Save the extracted mesh as an .obj file
def save_mesh_as_obj(vertices, triangles, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for t in triangles:
            f.write(f'f {t[0]+1} {t[1]+1} {t[2]+1}\n')

# Define the grid and evaluate the model
N = 200  # Resolution of the grid
xmin, xmax = -0.5, 0.5
ymin, ymax = -0.5, 0.5
zmin, zmax = -0.5, 0.5

grid = create_grid(N, xmin, xmax, ymin, ymax, zmin, zmax)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = OccupancyNetwork()
checkpoint = torch.load("/nas_data/data/wclw/ShadowArt_Occupency_Network/outcomes/outcome30.pth")
model.load_state_dict(checkpoint["model.state_dict"])
model.to(device)

occupancy_values = evaluate_occupancy_network(model, grid, device)

# Extract the mesh
vertices, triangles = extract_mesh(occupancy_values, grid)

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
outcome_dir = os.path.join(current_dir, 'outcomes')

# 创建 outcomes 文件夹（如果不存在）
if not os.path.exists(outcome_dir):
    os.makedirs(outcome_dir)

# 保存的临时 mesh 文件名
temp_mesh_filename = os.path.join(outcome_dir, 'A_temp_output_mesh.obj')
save_mesh_as_obj(vertices, triangles, temp_mesh_filename)
