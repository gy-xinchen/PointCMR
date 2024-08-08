# -*- coding: utf-8 -*-
# @Time    : 2024/5/27 10:19
# @Author  : yuan
# @File    : 3DMeshShow.py
import open3d as o3d
import numpy as np
import os

"""
    show motion point cloud
"""

# read point cloud from txt
def load_point_cloud_from_txt(txt_path):
    pcd = np.genfromtxt(txt_path, delimiter=" ")
    pcd_vector = o3d.geometry.PointCloud()
    pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])
    return pcd_vector

# get txt in file
def get_all_txt_files(folder_path):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    txt_files.sort()  
    return txt_files


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

    
    load_next_cloud(vis)

    # use N to next cloud
    vis.register_key_callback(ord("N"), load_next_cloud)

    vis.run()
    vis.destroy_window()


folder_path = r"...\Subject_0\motion"

display_point_cloud(folder_path)
