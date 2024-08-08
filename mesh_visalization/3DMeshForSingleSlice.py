import open3d as o3d
import numpy as np

"""
  to show point cloud [x,y,z]
"""

TxtPath = r"...\Subject_0\motion\RV_fr00.txt"

# read point cloud
pcd = np.genfromtxt(TxtPath, delimiter=" ")

pcd_vector = o3d.geometry.PointCloud()

pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])
o3d.visualization.draw_geometries([pcd_vector])



