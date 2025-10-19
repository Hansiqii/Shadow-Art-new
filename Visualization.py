from matplotlib import colors as mcolors  # 从matplotlib.colors导入
import numpy as np
import torch
import pyvista as pv
pv.OFF_SCREEN = True
from occupancy_network import OccupancyNetwork
import os


# Step 1: Create the 3D grid
def create_grid(N, xmin, xmax, ymin, ymax, zmin, zmax):
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)
    grid = np.stack(np.meshgrid(x, y, z), -1)
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


# Step 3: Visualize occupancy values using PyVista for interactivity
def visualize_occupancy_values(grid, occupancy_values, threshold=1e-1):
    # Flatten the grid and occupancy values
    grid_points = grid.reshape(-1, 3)
    occupancy_values_flat = occupancy_values.flatten()

    # Filter points where occupancy values are greater than the threshold
    valid_indices = occupancy_values_flat > threshold
    grid_points_filtered = grid_points[valid_indices]
    occupancy_values_filtered = occupancy_values_flat[valid_indices]

    # Create a PyVista point cloud with only valid points
    point_cloud = pv.PolyData(grid_points_filtered)

    # Map occupancy values to colors using PyVista's built-in colormap
    point_cloud.point_data["occupancy"] = occupancy_values_filtered

    # # Define a custom colormap from white to dark color (e.g., dark blue)
    cdict = {
        'red': [(0.0, 136 / 255, 136 / 255), (1.0, 255 / 255, 255 / 255)],  # From light blue to yellow (Red channel)
        'green': [(0.0, 213 / 255, 213 / 255), (1.0, 255 / 255, 255 / 255)],
        # From light blue to yellow (Green channel)
        'blue': [(0.0, 255 / 255, 255 / 255), (1.0, 0 / 255, 0 / 255)],  # From light blue to yellow (Blue channel)
    }

    # Create a custom colormap using the dictionary
    custom_cmap = mcolors.LinearSegmentedColormap('white_to_dark', cdict)
    # colormap = "spectral"

    # Apply the custom colormap
    point_cloud.point_data["occupancy"] = occupancy_values_filtered
    point_cloud.point_data["transparency"] = 1.0 - occupancy_values_filtered  # Map 0 -> 1 to 1 -> 0

    # Plot the point cloud with a color map and transparency
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud,
                     cmap= custom_cmap,
                     point_size=4,
                     render_points_as_spheres=True,
                     opacity="transparency",
                     scalar_bar_args={"title": None}
                     )

    plotter.set_background("white")
    plotter.show(screenshot="occupancy_output.png")
    print("✅ 保存完成：occupancy_output.png")


# Define grid size and range
N = 170  # Resolution of the grid
xmin, xmax = -0.5, 0.5
ymin, ymax = -0.5, 0.5
zmin, zmax = -0.5, 0.5

# Create grid
grid = create_grid(N, xmin, xmax, ymin, ymax, zmin, zmax)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = OccupancyNetwork()
checkpoint = torch.load(
    "/home/hsq/CODE/2510_FinalDesign/init_code/ShadowNet/outcomes/outcome30.pth")
model.load_state_dict(checkpoint["model.state_dict"])
model.to(device)

# Evaluate the occupancy network
occupancy_values = evaluate_occupancy_network(model, grid, device)

# Visualize the result, only showing points with occupancy_value > 1e-1
visualize_occupancy_values(grid, occupancy_values, threshold=1e-1)
