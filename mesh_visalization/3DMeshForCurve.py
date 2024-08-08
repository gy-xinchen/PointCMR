import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

"""
    data [x,y,z,intensity]
    to show 4D point cloud heart curve
"""

# read txt for point cloud
def load_point_cloud_from_txt(txt_path):
    data = np.loadtxt(txt_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])  # point [x, y, z]

    # use speed information to color code
    speeds = data[:, 3]
    norm_speeds = (speeds - speeds.min()) / (speeds.max() - speeds.min())  # nrom
    cmap = plt.get_cmap("jet")  # use jet map
    colors = cmap(norm_speeds)[:, :3] 
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


txt_path = r'...\Subject_171\rv_es_curvature.txt'
txt_path2 = r'...\Subject_172\rv_es_curvature.txt'

# load point cloud
point_cloud = load_point_cloud_from_txt(txt_path)
point_cloud2 = load_point_cloud_from_txt(txt_path2)

# visualization
o3d.visualization.draw_geometries([point_cloud])
o3d.visualization.draw_geometries([point_cloud2])
