# Neural Shadow Art

## Overview
This project implements a pipeline for image processing, light field optimization, registration, and 3D mesh reconstruction using neural representations.  
You can run the full pipeline with `python main.py`. The system allows you to optimize the light field, register input images, reconstruct 3D meshes, and visualize results.

---

## Data Input

Place your image data in the following directories:

- `data/Figures/`
- `data/input/`

Name the image files using indices `0-n`.  
It is crucial to maintain consistent numbering; otherwise, the program may read the images in the wrong order.

---

## Project Structure & Functionality

### 1. `train.py/generate_angle_screen()`
- **Purpose:** Controls whether the light field is optimized.  
- **Functionality:**  
  - Users can set the initial light field directions.  
  - Determines which angles of the light field are used for optimization.

### 2. `train_model_functions.py`
- **Purpose:** Contains helper functions for training the neural network.  
- **Key function:**
  - `truncate_ray`: Manually calculates and controls whether rays are truncated during training, affecting the accuracy and stability of the reconstruction.

### 3. `registration.py`
- **Purpose:** Implements registration functions.  
- **Functionality:**  
  - Aligns input images and/or reconstructed 3D structures.  
  - Ensures consistent coordinate frames for further processing.

### 4. `reconstruction.py`
- **Purpose:** Reconstructs 3D meshes from optimized light fields or registered images.  
- **Key parameters:**  
  - Mesh resolution is set to 200 by default.

### 5. `painter.py`
- **Purpose:** Handles projection rendering.  
- **Functionality:**  
  - Converts 3D meshes or neural representations into 2D projections.  
  - Can be used for visualization or comparison with input images.

### 6. `visualization.py`
- **Purpose:** Observes and inspects neural representation values.  
- **Functionality:**  
  - Helps debug and understand the internal state of the neural network.  
  - Provides numerical outputs that can be visualized or analyzed.

### 7. `diff.py`
- **Purpose:** Computes various image differences.  
- **Functionality:**  
  - Measures error between projected images and target/reference images.  
  - Supports multiple metrics for evaluating reconstruction quality.

---

## Usage

1. Place input images in `data/Figures/` and `data/input/` with numbering `0-n`.  
2. Optionally configure `generate_angle_screen.py` to optimize the light field and set initial directions.  
3. Adjust `truncate_ray` in `train_model_functions.py` if you want to truncate rays manually.  
4. Run the pipeline:
python main.py

**Note:** If the initial few losses do not decrease quickly, you may need to try running the code again; otherwise, the optimization may not converge. Running the code multiple times usually ensures convergence.

