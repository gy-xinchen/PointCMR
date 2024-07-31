# -*- coding: utf-8 -*-
# @Time    : 2024/5/27 10:19
# @Author  : yuan
# @File    : 3DMeshShow.py
import open3d as o3d
import numpy as np
import os

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

# 显示点云
def display_point_cloud(folder_path):
    txt_files = get_all_txt_files(folder_path)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    def load_next_cloud(vis):
        nonlocal txt_files
        if len(txt_files) == 0:
            return
        txt_file = txt_files.pop(0)
        pcd = load_point_cloud_from_txt(os.path.join(folder_path, txt_file))
        vis.clear_geometries()
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    # 加载并显示第一个点云文件
    load_next_cloud(vis)

    # 绑定按键回调函数
    vis.register_key_callback(ord("N"), load_next_cloud)

    vis.run()
    vis.destroy_window()

# 使用指定路径显示点云
folder_path = r"F:\袁心辰研究生资料\4DPH_anaylsis\4Dsegment-master\all_mPAP_4dnii_data0_60\Subject_0\motion"

display_point_cloud(folder_path)
