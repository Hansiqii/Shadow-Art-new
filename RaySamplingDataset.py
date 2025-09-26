import torch
import numpy as np
from torch.utils.data import Dataset
from data_preprocessing import load_images_and_angles
from torch.utils.data import DataLoader
import os
import cv2

class RaySamplingDataset(Dataset):
    """A dataset generating ray sample coordinates and corresponding occupancy values.
    Parameters:
        img_dir:            The directory of binary images of the same dimension that correspond
                            to silhouettes, with 1 WHITE being unoccupied and 0 BLACK being occupied.
        angles_file_path:   The path to a CSV file that has ANGLE_RAD as its first column.
                            The column ANGLE_RAD has a planar viewing angle in radians, e.g. 3.14,
                            separated by newlines."""

    def __init__(self, device, img_dir, angles, screens, n=100):  # Added max_ray_length as a parameter
        # Get a list of 2-tuples, each 2-tuple contains a numpy 2D object and a float; integer; integer
        self.images_and_angles, self.height, self.width = load_images_and_angles(img_dir, angles)
        self.screens = screens
        self.n = n
        self.device = device
        # Some checks
        for img_idx in range(len(self.images_and_angles)):
            # Transform the images to contain occupancy values instead, store the result in a torch tensor
            self.images_and_angles[img_idx] = (
            torch.tensor((255 - self.images_and_angles[img_idx][0])/255, dtype=torch.float32),
            self.images_and_angles[img_idx][1])

            # Check whether the dimensions are correct
            if not self.images_and_angles[img_idx][0].shape == (self.height, self.width):
                raise ValueError(f"The dimensions of file {img_idx} are not {self.height} X {self.width}")

        print(self)

    def __str__(self):
        return (f"Successfully initialised the RaySamplingDataset with:\n\
              \tLoaded images:\t\t {len(self.images_and_angles)}\n\
              \tAt angles(RAD):\t\t{[self.images_and_angles[i][1] for i in range(len(self.images_and_angles))]}\n\
              \tn:\t\t{self.n}\n\
              \t(width, height):\t\t({self.width}, {self.height})")

    def __len__(self):  # Need to implement.
        # The total number of pixels in the images combined.
        return len(self.images_and_angles) * self.height * self.width

    def __getitem__(self, idx):  # Need to reimplement
        """Samples array of points, with a parameter n which cannot be altered directly. It samples a
        pixel value from one of the (three) input images and generates the query array as a torch (n-1, 3)
        tensor object. It then returns this object and its corresponding occupancy value."""

        # Find the corresponding pixel coordinates.
        r, c = idx % (self.height * self.width) // self.width, idx % (self.height * self.width) % self.width
        # print(f"Getting r,c={r},{c}, 0<=r<{self.height}, c range is 0<=c<{self.width}")

        # Load its binary occupancy value.
        img_idx = idx // (self.height * self.width)
        pixel_val = self.images_and_angles[img_idx][0][r, c]

        # Load its angle.
        angle = self.images_and_angles[img_idx][1]
        screen = self.screens[img_idx]
        # Generate a sampling ray using these values
        ray = self.generate_ray(r, c, angle, screen, self.n, img_idx)

        # calculate volume weight
        differences = ray[1:] - ray[:-1]
        pre_volume = torch.norm(differences, dim=1)
        volume = torch.cat((pre_volume[:1], (pre_volume[:-1] + pre_volume[1:]) / 2, pre_volume[-1:]), dim=0)
        return ray.to(self.device), pixel_val, volume.to(self.device), img_idx, r, c

    # def generate_sampling_ray_splits(self, r, c, angle, n, width_3d_rectangle=2, distance_from_center=1.0):
    #     """Returns a torch tensor object with shape (n,3)"""
    #
    #     # Normalize the angle vector
    #     processed_angle = []
    #
    #     # 遍历 angle 列表中的每个元素
    #     for component in angle:
    #         if isinstance(component, torch.Tensor):  # 如果是张量
    #             processed_angle.append(component.detach().item())  # 转为标量
    #         else:  # 如果是普通数值
    #             processed_angle.append(float(component))  # 确保是浮点数
    #
    #     # 将处理后的角度列表转换为 NumPy 数组
    #     processed_angle_np = np.array(processed_angle)
    #     norm = np.linalg.norm(processed_angle_np)
    #     processed_angle_np = processed_angle_np / norm
    #
    #     # Calculate the camera center position in 3D space
    #     x0, y0, z0 = processed_angle_np * distance_from_center
    #     image_center = torch.tensor([x0, y0, z0], dtype=torch.float32, device=self.device)
    #
    #     # Calculate the aspect ratio
    #     aspect_ratio = self.width * 1.0 / self.height
    #
    #     # Calculate c_unit and normalize it
    #     if processed_angle_np[1] == 0 and processed_angle_np[0] == 0:
    #         c_unit = np.array([0, 1, 0])
    #     else:
    #         temp_vec = np.array([-processed_angle_np[1], processed_angle_np[0], 0])
    #         c_unit = temp_vec / np.linalg.norm(temp_vec)
    #
    #     # Calculate r_unit
    #     r_unit = np.cross(c_unit, processed_angle_np)
    #
    #     # Convert c_unit and r_unit to torch tensors
    #     c_unit = torch.tensor(c_unit, dtype=torch.float32, device=self.device)
    #     r_unit = torch.tensor(r_unit, dtype=torch.float32, device=self.device)
    #
    #     # Calculate the start position of the image plane
    #     image_start = image_center - width_3d_rectangle * 0.5 * c_unit - width_3d_rectangle / aspect_ratio * 0.5 * r_unit
    #
    #     # Calculate the specific pixel position on the image plane
    #     image_rc = image_start + width_3d_rectangle * c / self.width * c_unit + width_3d_rectangle / aspect_ratio * r / self.height * r_unit
    #
    #     # Allocate space for the ray
    #     ray = torch.zeros(n, 3, dtype=torch.float32)
    #
    #     # Assign the dimensions one by one
    #     torch.linspace(image_rc[0], image_rc[0] - 2 * image_center[0], n, out=ray[:, 0])
    #     torch.linspace(image_rc[1], image_rc[1] - 2 * image_center[1], n, out=ray[:, 1])
    #     torch.linspace(image_rc[2], image_rc[2] - 2 * image_center[2], n, out=ray[:, 2])
    #
    #     return ray

    def generate_sampling_ray_splits(self, r, c, angle, screen, n, img_idx, width_3d_rectangle=1, distance_from_center=0.5):
        """Returns a torch tensor object with shape (n, 3)"""

        # Normalize the angle/screen vector
        angle = angle.to(self.device)
        norm_angle = torch.norm(angle)
        angle = angle / norm_angle

        screen = screen.to(self.device)
        norm_screen = torch.norm(screen)
        screen = screen / norm_screen

        # calculate method 1
        fix_distance = distance_from_center/ torch.dot(screen, angle)
        image_center = -angle * fix_distance

        # # calculate method 2
        # fix_width = width_3d_rectangle * torch.dot(screen, angle)
        # image_center = -angle * distance_from_center

        # Calculate the aspect ratio
        aspect_ratio = self.width * 1.0 / self.height

        if screen[1] == 0 and screen[0] == 0:
            c_unit = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
        else:
            temp_vec = torch.empty(3).to(self.device)
            temp_vec[0] = -screen[1]  # 第二个行，第一列
            temp_vec[1] = screen[0]# 第二个行，第二列
            temp_vec[2] = 0.0  # 第二个行，第三列
            c_unit = temp_vec / torch.norm(temp_vec)

        # calculate method 1
        r_unit = torch.cross(c_unit, screen, dim=0)
        image_start = image_center - width_3d_rectangle * 0.5 * c_unit - width_3d_rectangle / aspect_ratio * 0.5 * r_unit
        image_rc = image_start + width_3d_rectangle * c / self.width * c_unit + width_3d_rectangle / aspect_ratio * r / self.height * r_unit
        # # calculate method 2
        # c_unit = (c_unit + torch.dot(-c_unit,angle) * angle) / torch.norm(c_unit - torch.dot(c_unit,angle) * angle)
        # r_unit = torch.cross(c_unit, angle, dim=0)
        # image_start = image_center - fix_width * 0.5 * c_unit - fix_width / aspect_ratio * 0.5 * r_unit
        # image_rc = image_start + fix_width * c / self.width * c_unit + fix_width / aspect_ratio * r / self.height * r_unit


        # calculation method 1
        start = image_rc[0]
        end = image_rc[0] + 2 * fix_distance * angle[0]
        ray_0 = start + (end - start) * torch.linspace(0, 1, n, device=self.device).requires_grad_()
        start = image_rc[1]
        end = image_rc[1] + 2 * fix_distance * angle[1]
        ray_1 = start + (end - start) * torch.linspace(0, 1, n, device=self.device).requires_grad_()
        start = image_rc[2]
        end = image_rc[2] + 2 * fix_distance * angle[2]
        ray_2 = start + (end - start) * torch.linspace(0, 1, n, device=self.device).requires_grad_()

        # # calculation method 2
        # start = image_rc[0]
        # end = image_rc[0] + 2 * distance_from_center * angle[0]
        # ray_0 = start + (end - start) * torch.linspace(0, 1, n, device=self.device).requires_grad_()
        # start = image_rc[1]
        # end = image_rc[1] + 2 * distance_from_center * angle[1]
        # ray_1 = start + (end - start) * torch.linspace(0, 1, n, device=self.device).requires_grad_()
        # start = image_rc[2]
        # end = image_rc[2] + 2 * distance_from_center * angle[2]
        # ray_2 = start + (end - start) * torch.linspace(0, 1, n, device=self.device).requires_grad_()

        # 不进行原地操作，构建新的 ray 张量
        ray = torch.stack([ray_0, ray_1, ray_2], dim=-1)

        return ray

    def generate_sampled_ray(self, ray_splits):
        """Returns a ray of uniformly sampled 3D points within the line pieces defined by ray_splits.
        Params:
                ray_splits  torch.tensor object with shape (n,3), n>=1, dtype torch.float32.
        Returns:
                sampled_ray torch.tensor object with shape (n-1, 3), dtype torch.float32"""
        differences = ray_splits[1:, :] - ray_splits[:-1, :]
        coordinates = torch.zeros(size=(differences.shape[0],), dtype=torch.float32).uniform_().to(self.device)
        return ray_splits[:-1, :] + (differences.T * coordinates).T

    def generate_ray(self, r, c, angle, screen, n, img_idx):
        """Generates a ray."""
        ray_splits = self.generate_sampling_ray_splits(r, c, angle, screen, n, img_idx)
        return self.generate_sampled_ray(ray_splits)

    def area_correction(self,img_dir):
        max_area = 0

        # 遍历目录中的所有文件
        for filename in os.listdir(img_dir):
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                file_path = os.path.join(img_dir, filename)
                # 读取图像
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                # 二值化图像
                binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
                # 计算当前图像的最小矩形面积
                area = self.find_min_rectangle_area(binary_image)
                # 更新最大面积
                max_area = max(max_area, area)

        return self.width * self.height / max_area

    def find_min_rectangle_area(self, binary_image):
        # 找到所有轮廓
        binary_image = cv2.bitwise_not(binary_image)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 如果没有找到任何轮廓，返回面积为0
        if not contours:
            return 0

        # 初始化边界框的最小和最大值
        x, y, w, h = cv2.boundingRect(contours[0])
        max_x = x + w
        max_y = y + h
        min_x = x
        min_y = y

        # 获取所有轮廓的边界框的并集
        for cnt in contours[1:]:
            x2, y2, w2, h2 = cv2.boundingRect(cnt)
            max_x = max(max_x, x2 + w2)
            max_y = max(max_y, y2 + h2)
            min_x = min(min_x, x2)
            min_y = min(min_y, y2)
            w = max_x - min_x
            h = max_y - min_y

        # 计算最小矩形的面积
        area = w * h

        return area

# current_dir = os.path.dirname(os.path.abspath(__file__))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data_dir = os.path.join(current_dir, "data\\Figures")
# angles_file = os.path.join(current_dir, "data\\angle.txt")
# dataset = RaySamplingDataset(device, data_dir, angles_file)
# print(dataset.__getitem__(79*100+72))

