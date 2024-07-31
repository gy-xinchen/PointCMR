import open3d as o3d
import numpy as np

TxtPath = r"F:\袁心辰研究生资料\4DPH_anaylsis\4Dsegment-master\all_mPAP_4dnii_data0_60\Subject_0\motion\RV_fr00.txt"

# 通过numpy读取txt点云
pcd = np.genfromtxt(TxtPath, delimiter=" ")

pcd_vector = o3d.geometry.PointCloud()
# 加载点坐标
pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])
o3d.visualization.draw_geometries([pcd_vector])



