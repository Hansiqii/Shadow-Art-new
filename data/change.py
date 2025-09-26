import cv2
import os


# 定义处理图片的函数
def invert_images_in_current_folder():
    # 获取程序当前所在的文件夹路径
    current_folder = os.getcwd()

    # 创建一个保存翻转图片的文件夹
    save_folder = os.path.join(current_folder, 'inverted_images')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 遍历当前文件夹中的所有文件
    for filename in os.listdir(current_folder):
        # 检查文件是否是图片文件
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # 构建图片的完整路径
            img_path = os.path.join(current_folder, filename)

            # 读取图片，确保以灰度模式读取
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # 如果图片成功读取
            if image is not None:
                # 进行黑白翻转
                inverted_image = cv2.bitwise_not(image)

                # 构建保存图片的完整路径
                save_path = os.path.join(save_folder, filename)

                # 保存翻转后的图片
                cv2.imwrite(save_path, inverted_image)
                print(f"{filename} 翻转并保存到 {save_path}")
            else:
                print(f"{filename} 无法读取！")


# 调用函数处理图片
invert_images_in_current_folder()
