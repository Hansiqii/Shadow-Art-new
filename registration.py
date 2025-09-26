import open3d as o3d
import numpy as np
import cv2
import os
from PIL import Image

# Step 1: 提取图像边缘并生成3D点云
def extract_edges_and_point_cloud(binary_image):
    edges = cv2.Canny(binary_image, 100, 200)
    points = np.column_stack(np.where(edges > 0))
    #points = np.column_stack(np.where(binary_image > 0))
    points_3d = np.hstack([points, np.zeros((points.shape[0], 1))])
    return points_3d

# Step 2: 使用Open3D将边缘点转换为点云格式
def create_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def registration(current_dir ,image_1, image_2, epoch, img_idx):
    imageA = cv2.bitwise_not(cv2.imread(image_1, cv2.IMREAD_GRAYSCALE))
    imageB = cv2.bitwise_not(cv2.imread(image_2, cv2.IMREAD_GRAYSCALE))

    pointsA = extract_edges_and_point_cloud(imageA)
    pointsB = extract_edges_and_point_cloud(imageB)

    pcdA = create_point_cloud(pointsA)
    pcdB = create_point_cloud(pointsB)

# Step 4: ICP 配准
    threshold = 100  # 定义ICP算法的距离阈值
    trans_init = np.eye(4)  # 初始变换矩阵
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcdA, pcdB, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print("ICP Transformation:\n", reg_p2p.transformation)

# Step 5: 提取2D变换矩阵
    transformation_matrix = reg_p2p.transformation[:2, [0, 1, 3]]

# Step 6: 应用变换矩阵到整个图像A
    rows, cols = imageA.shape
    pre_image = cv2.warpAffine(imageA, transformation_matrix, (cols, rows))
    contours, _ = cv2.findContours(pre_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 提取所有黑色区域轮廓点
    black_points = np.vstack(contours).squeeze()  # 提取所有轮廓点并压缩维度
    # 对黑色区域的轮廓点进行变换
    transformed_black_points = cv2.transform(np.array([black_points], dtype='float32'), transformation_matrix)[0]

    # 检查黑色区域变换后是否出界
    if (transformed_black_points[:, 0] >= 0).all() and (transformed_black_points[:, 0] < cols).all() and \
            (transformed_black_points[:, 1] >= 0).all() and (transformed_black_points[:, 1] < rows).all():
        # 如果黑色区域未出界，执行变换
        warped_imageA = cv2.bitwise_not(cv2.warpAffine(imageA, transformation_matrix, (cols, rows)))
        _, warped_imageA = cv2.threshold(warped_imageA, 127, 255, cv2.THRESH_BINARY)
    else:
        # 如果出界，不进行变换，保持原图
        warped_imageA = cv2.bitwise_not(imageA)
        _, warped_imageA = cv2.threshold(warped_imageA, 127, 255, cv2.THRESH_BINARY)


# Step 7: 保存或显示变换后的图像
    output_dir = os.path.join(current_dir, 'outcomes')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(current_dir, f"outcomes/Registration_Figure{img_idx}_Epoch{epoch+1}.png"), warped_imageA)
    cv2.imwrite(os.path.join(current_dir, f"data/Figures/{img_idx}.png"), warped_imageA)
# icp_pcd = deepcopy(pcdA)
# icp_pcd.transform(reg_p2p.transformation)
#
# pcdA.paint_uniform_color([0, 1, 0])
# pcdB.paint_uniform_color([1, 0, 0])
# Step 4: 可视化配准前的点云
# print("显示配准前的点云")
# o3d.visualization.draw_geometries([pcdA, pcdB,icp_pcd])
# current_dir = os.path.dirname(os.path.abspath(__file__))
# registration(current_dir, os.path.join(current_dir, f"data/Figures/{0}.png"), os.path.join(current_dir, f"data/temp/outcome_{0}.png"), 114, 0)



