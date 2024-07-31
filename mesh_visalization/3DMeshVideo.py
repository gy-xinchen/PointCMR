import open3d as o3d
import numpy as np
import os
from PIL import Image

def rotate_pc_along_x_axis(pcd, angle_degrees):
    """绕X轴旋转点云"""
    R = pcd.get_rotation_matrix_from_xyz((0, np.radians(angle_degrees), 0))
    pcd.rotate(R, center=(0, 0, 0))
    return pcd

def rotate_pc_along_y_axis(pcd, angle_degrees):
    """绕Y轴旋转点云"""
    R = pcd.get_rotation_matrix_from_xyz((np.radians(angle_degrees), 0, 0))
    pcd.rotate(R, center=(0, 0, 0))
    return pcd

def rotate_pc_along_z_axis(pcd, angle_degrees):
    """绕Z轴旋转点云"""
    R = pcd.get_rotation_matrix_from_xyz((0, 0, np.radians(angle_degrees)))
    pcd.rotate(R, center=(0, 0, 0))
    return pcd


# 读取txt文件并生成点云
def load_point_cloud_from_txt(txt_path):
    pcd = np.genfromtxt(txt_path, delimiter=" ")
    pcd_vector = o3d.geometry.PointCloud()
    pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])
    return pcd_vector


# 获取文件夹中所有的txt文件
def get_all_txt_files(folder_path):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    txt_files.sort()  # 排序文件以确保顺序正确
    return txt_files


# 生成并保存GIF
def generate_point_cloud_gif(folder_path, output_gif_path, fps=10):
    txt_files = get_all_txt_files(folder_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0, 0, 0])

    images = []
    for txt_file in txt_files:
        pcd = load_point_cloud_from_txt(os.path.join(folder_path, txt_file))
        pcd_rotated = rotate_pc_along_x_axis(pcd, 90)
        pcd_rotated = rotate_pc_along_z_axis(pcd_rotated,90)

        vis.clear_geometries()
        vis.add_geometry(pcd_rotated)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image("temp.png")

        # 打开并关闭图像文件，以便稍后删除
        img = Image.open("temp.png")
        images.append(img.copy())
        img.close()

    vis.destroy_window()
    os.remove("temp.png")

    # 保存为GIF文件
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)


# 使用指定路径生成GIF
folder_path = r"F:\袁心辰研究生资料\4DPH_anaylsis\all_mPAP_4dnii_data171_258\Subject_171\motion"
output_gif_path = r"F:\袁心辰研究生资料\4DPH_anaylsis\all_mPAP_4dnii_data171_258\Subject_171\point_cloud_animation171.gif"
generate_point_cloud_gif(folder_path, output_gif_path)
