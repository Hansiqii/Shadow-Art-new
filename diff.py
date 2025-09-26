import cv2
import numpy as np
import os

def compute_diff(current_dir, image_1, image_2, epoch, img_idx, bool):
    A = cv2.imread(image_1,  cv2.IMREAD_GRAYSCALE)
    B = cv2.imread(image_2, cv2.IMREAD_GRAYSCALE)

# 将A转换为三通道RGB图像
    output = cv2.cvtColor(A, cv2.COLOR_GRAY2BGR)

# 设置条件
# A有，B没有 -> 蓝色 (0, 0, 255)
    output[(A == 255) & (B == 0)] = [255, 0, 0]

# A没有，B有 -> 橙色 (255, 165, 0)
    output[(A == 0) & (B == 255)] = [0, 165, 255]

# 保存结果
    if bool == 1:
        cv2.imwrite(os.path.join(current_dir, f"outcomes/RegistrationDiff{img_idx}_Epoch{epoch + 1}.png"), output)
    elif bool == 0:
        cv2.imwrite(os.path.join(current_dir, f"outcomes/FiguresDiff{img_idx}_Epoch{epoch + 1}.png"), output)
    elif bool == 2:
        cv2.imwrite(os.path.join(current_dir, f"outcomes/FiguresRegDiff{img_idx}_Epoch{epoch + 1}.png"), output)

# current_dir = os.path.dirname(os.path.abspath(__file__))
# compute_diff(current_dir,
#              "D:/USTC_CC/ShadowArt with Occupency Network/ShadowArt_Occupency_Network/data/Figures/0.png", "D:/USTC_CC/ShadowArt with Occupency Network/ShadowArt_Occupency_Network/9.18/outcome_Figure0_Epoch20.png",114, 0, 1)