import cv2
import numpy as np
import torch

def find_min_rectangle_area(binary_image):
    # 找到所有轮廓
    binary_image = cv2.bitwise_not(binary_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有找到任何轮廓，返回面积为0
    if not contours:
        return 0

    # 初始化边界框的最小和最大值
    x, y, w, h = cv2.boundingRect(contours[0])
    max_x = x+w
    max_y = y+h
    min_x = x
    min_y = y

    # 获取所有轮廓的边界框的并集
    for cnt in contours[1:]:
        x2, y2, w2, h2 = cv2.boundingRect(cnt)
        max_x = max(max_x, x2+w2)
        max_y = max(max_y, y2+h2)
        min_x = min(min_x,x2)
        min_y = min(min_y,y2)
        w = max_x - min_x
        h = max_y - min_y

    # 计算最小矩形的面积
    print(w,h)
    area = w * h

    return area

# 测试用例
binary_image = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)[1]  # 二值化
area = find_min_rectangle_area(binary_image)
print("最小正方形的像素面积为:", area)
