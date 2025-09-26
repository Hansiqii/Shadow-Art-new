import os
import cv2
import numpy as np

from train import process_images
def calculate_iou_and_ds(img1, img2):
    """
    计算两张图片的 IOU 和 Dice Similarity。
    """
    if img1.shape != img2.shape:
        raise ValueError("输入的两张图片尺寸必须相同")

    intersection = np.logical_and(img1, img2).sum()
    union = np.logical_or(img1, img2).sum()
    iou = intersection / union if union != 0 else 0.0
    ds = (2 * intersection) / (img1.sum() + img2.sum()) if (img1.sum() + img2.sum()) != 0 else 0.0

    return iou, ds


def preprocess_and_calculate(img_path1, img_path2, threshold=127):
    """
    读取图片、二值化并计算 IOU 和 DS。
    """
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError(f"无法加载图片: {img_path1} 或 {img_path2}")

    # 确保图片大小一致
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 二值化处理
    _, binary_img1 = cv2.threshold(img1, threshold, 1, cv2.THRESH_BINARY)
    _, binary_img2 = cv2.threshold(img2, threshold, 1, cv2.THRESH_BINARY)

    return calculate_iou_and_ds(binary_img1, binary_img2)


def calculate_for_folders(folder1, folder2, threshold=127):
    """
    对两个文件夹中相同名称的图片计算 IOU 和 DS。
    """
    # 列出两个文件夹中的文件
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # 找出两个文件夹中共同的文件
    common_files = files1.intersection(files2)

    if not common_files:
        print("两个文件夹中没有名称相同的图片")
        return

    results = []

    for file_name in common_files:
        img_path1 = os.path.join(folder1, file_name)
        img_path2 = os.path.join(folder2, file_name)

        try:
            iou, ds = preprocess_and_calculate(img_path1, img_path2, threshold)
            results.append((file_name, iou, ds))
            print(f"{file_name}: IOU = {iou:.3f}, DS = {ds:.3f}")
        except Exception as e:
            print(f"处理 {file_name} 时出错: {e}")

    return results


# 示例：输入文件夹路径
folder1 = "D:\\USTC_CC\\ShadowArt with Occupency Network\\ShadowArt_Occupency_Network\\data\\animals"
folder2 = "D:\\USTC_CC\\ShadowArt with Occupency Network\\ShadowArt_Occupency_Network\\data\\Root"

process_images(folder1, size=(50, 50))
process_images(folder2, size=(50, 50))
# 执行计算
results = calculate_for_folders(folder1, folder2, threshold=127)

# 保存结果到文件（可选）
output_file = "results.txt"
with open(output_file, "w") as f:
    for file_name, iou, ds in results:
        f.write(f"{file_name}: IOU = {iou:.3f}, DS = {ds:.3f}\n")

print(f"结果已保存到 {output_file}")

