import os
import cv2

def convert_to_binary_image(image):
    """
    将图像转换为二进制图像
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return binary_image

def process_images_in_directory_recursive(directory):
    """
    递归处理指定目录下的所有子文件夹中的RGB图像，并保存为二进制PNG图像
    """
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"无法读取图像文件 {img_path}")
                    continue

                binary_image = convert_to_binary_image(image)

                # 输出文件路径：保持原子目录结构
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(root, output_filename)

                cv2.imwrite(output_path, binary_image)
                print(f"Updated binary image: {output_path}")

# 使用示例
# if __name__ == "__main__":
#     input_directory = "D:\\USTC_CC\\ShadowArt with Occupency Network\\ShadowArt_Occupency_Network\\data"  # 替换为主文件夹路径
#     process_images_in_directory_recursive(input_directory)
