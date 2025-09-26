import cv2
import os

def flip_image(image):
    """
    对图像进行水平翻转
    """
    return cv2.flip(image, 1)

def process_images_in_directory(directory):
    """
    处理指定目录下的所有图像，并保存为水平翻转后的PNG图像
    """
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            print(f"正在处理图像: {img_path}")

            if not os.path.exists(img_path):
                print(f"文件路径不存在: {img_path}")
                continue

            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"无法读取图像文件 {img_path}")
                    continue

                flipped_image = flip_image(image)
                output_filename = os.path.splitext(filename)[0] + '_flipped.png'
                output_path = os.path.join(directory, output_filename)

                cv2.imwrite(output_path, flipped_image)
                print(f"已保存对称图像: {output_path}")

            except Exception as e:
                print(f"处理文件 {img_path} 时发生错误: {e}")

if __name__ == "__main__":
    current_directory = os.getcwd()
    print(f"当前工作目录: {current_directory}")
    process_images_in_directory("D:\\USTC_CC\\ShadowArt with Occupency Network\\ShadowArt_Occupency_Network\\data\\Figures")
