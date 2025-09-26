import cv2
import os


# 定义处理图片的函数
def invert_images_in_directory_recursive(directory):
    """
    递归处理目录及其子目录下的所有图像文件，进行黑白翻转并覆盖原图。
    """
    for root, _, files in os.walk(directory):
        for filename in files:
            # 检查文件是否是图片文件
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # 构建图片的完整路径
                img_path = os.path.join(root, filename)

                # 读取图片，确保以灰度模式读取
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # 如果图片成功读取
                if image is not None:
                    # 进行黑白翻转
                    inverted_image = cv2.bitwise_not(image)

                    # 直接覆盖保存翻转后的图片
                    cv2.imwrite(img_path, inverted_image)
                    print(f"{filename} 已翻转并覆盖原文件")
                else:
                    print(f"{filename} 无法读取！")


# 调用函数处理图片
if __name__ == "__main__":
    target_directory = "D:\\USTC_CC\\ShadowArt with Occupency Network\\ShadowArt_Occupency_Network\\data-inverse\\ShadowArt"  # 替换为目标文件夹路径
    invert_images_in_directory_recursive(target_directory)
