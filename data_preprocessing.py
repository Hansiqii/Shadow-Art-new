import numpy as np
import imageio.v2 as imageio
import os


def convert_to_binary(image, threshold=0.5):
    """Convert an image to a binary image"""
    print("hello!")
    if image.ndim == 3 and image.shape[2] == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    binary_image = (image > threshold).astype(np.uint8)

    return binary_image


def is_binary(image):
    """Check if an image is a binary image"""
    return np.all((image == 0) | (image == 255))

# def load_soft_angles(image_folder, angle_file):
#     with open(angle_file, 'r') as f:
#         angles = [[float(num) for num in line.split()] for line in f.readlines()]
#     image_names = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     print(f"Loaded: {image_names}")
#     images = [imageio.imread(os.path.join(image_folder, f)) for f in image_names]
#
#     # Check and convert to binary images
#     binary_images = []
#     for img in images:
#         if not is_binary(img):
#             binary_images.append(convert_to_binary(img))
#         else:
#             binary_images.append(img)
#
#     height, width = binary_images[0].shape[:2]
#     return angles, height, width

# def load_images_and_angles(image_folder, angle_file):
#     # Load all images from the image folder
#     image_names = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     print(f"Loaded: {image_names}")
#     images = [imageio.imread(os.path.join(image_folder, f)) for f in image_names]
#
#     # Check and convert to binary images
#     binary_images = []
#     for img in images:
#         if not is_binary(img):
#             print("Warning!")
#             binary_images.append(convert_to_binary(img))
#         else:
#             binary_images.append(img)
#
#     # Load the angle file
#     with open(angle_file, 'r') as f:
#         angles = [[float(num) for num in line.split()] for line in f.readlines()]
#
#     # Check if the number of images matches the number of angles
#     if len(binary_images) != len(angles):
#         raise ValueError("The number of images does not match the number of angles!")
#
#     # Pair images with their corresponding angles
#     image_angle_pairs = list(zip(binary_images, angles))
#
#     # Get the width and height of the first image
#     height, width = binary_images[0].shape[:2]
#
#     return image_angle_pairs, height, width

def load_images_and_angles(image_folder, angles):
    # Load all images from the image folder
    # image_names = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_names = [f"{i}.png" for i in range(len(os.listdir(image_folder)))]
    print(f"Loaded: {image_names}")
    images = [imageio.imread(os.path.join(image_folder, f)) for f in image_names]

    # Check and convert to binary images
    binary_images = []
    for img in images:
        if not is_binary(img):
            print("Warning!")
            binary_images.append(convert_to_binary(img))
        else:
            binary_images.append(img)

    # Load the angle file
    # with open(angle_file, 'r') as f:
    #     angles = [[float(num) for num in line.split()] for line in f.readlines()]

    # Check if the number of images matches the number of angles
    if len(binary_images) != len(angles):
        raise ValueError("The number of images does not match the number of angles!")

    # Pair images with their corresponding angles
    image_angle_pairs = list(zip(binary_images, angles))

    # Get the width and height of the first image
    height, width = binary_images[0].shape[:2]

    return image_angle_pairs, height, width

# load_images_and_angles("D:\\USTC_CC\\ShadowArt with Occupency Network\\ShadowArt_Occupency_Network\\data\\input",
#                        "D:\\USTC_CC\\ShadowArt with Occupency Network\\ShadowArt_Occupency_Network\\data\\angle.txt")
