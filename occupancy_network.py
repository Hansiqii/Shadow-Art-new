import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# Positional encoding function to increase the representational capacity of the input x
def positional_encoding(x, num_encoding_functions=6):
    encoding = [x]
    for i in range(num_encoding_functions):
        for func in [torch.sin, torch.cos]:
            encoding.append(func((2.0**i) * x))
    return torch.cat(encoding, dim=-1)


class OccupancyNetwork(nn.Module):
    def __init__(self, num_encoding_functions=6):
        super(OccupancyNetwork, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        input_dim = (
            3 + 3 * 2 * num_encoding_functions
        )  # 3 original dims + 2 (sin, cos) * 3 dims * num_encoding_functions

        # Define fully connected layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 1)

        # 定义可学习的角度/屏幕参数
        self.delta_y_1 = nn.Parameter(
            torch.tensor([0.0], requires_grad=True)
        )  # 初始化为 0.3
        self.delta_x_2 = nn.Parameter(
            torch.tensor([0.0], requires_grad=True)
        )  # 初始化为 0.5
        self.delta_z_1 = nn.Parameter(
            torch.tensor([0.0], requires_grad=True)
        )  # 初始化为 0.3
        self.delta_z_2 = nn.Parameter(
            torch.tensor([0.0], requires_grad=True)
        )  # 初始化为 0.5
        self.delta_x_3 = nn.Parameter(
            torch.tensor([0.0], requires_grad=True)
        )  # 初始化为 0.3
        self.delta_y_3 = nn.Parameter(
            torch.tensor([0.0], requires_grad=True)
        )  # 初始化为 0.5

        self.screen_y_1 = nn.Parameter(
            torch.tensor([0.0], requires_grad=True)
        )  # 初始化为 0.3
        self.screen_x_2 = nn.Parameter(
            torch.tensor([0.0], requires_grad=True)
        )  # 初始化为 0.5
        self.screen_z_1 = nn.Parameter(
            torch.tensor([0.0], requires_grad=True)
        )  # 初始化为 0.3
        self.screen_z_2 = nn.Parameter(
            torch.tensor([0.0], requires_grad=True)
        )  # 初始化为 0.5
        self.screen_x_3 = nn.Parameter(
            torch.tensor([0.0], requires_grad=True)
        )  # 初始化为 0.3
        self.screen_y_3 = nn.Parameter(
            torch.tensor([0.0], requires_grad=True)
        )  # 初始化为 0.5
        # Initialize network parameters to output 0.5 initially
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize all fully connected layers' weights to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)  # Set biases to zero
        # Specifically, set the bias of the last layer to 0 to ensure sigmoid outputs 0.5
        nn.init.constant_(self.fc8.bias, 0.0)

    def forward(self, x):
        x = positional_encoding(x, self.num_encoding_functions)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = torch.sigmoid(self.fc8(x))
        return x

    # def forward(self, x):
    #     x = positional_encoding(x, self.num_encoding_functions)  # 不涉及inplace
    #     x = F.relu(self.fc1(x), inplace=False)  # 禁用inplace
    #     x = F.relu(self.fc2(x), inplace=False)
    #     x = F.relu(self.fc3(x), inplace=False)
    #     x = F.relu(self.fc4(x), inplace=False)
    #     x = F.relu(self.fc5(x), inplace=False)
    #     x = F.relu(self.fc6(x), inplace=False)
    #     x = F.relu(self.fc7(x), inplace=False)
    #     x = torch.sigmoid(self.fc8(x))  # 这没有inplace操作
    #     return x
