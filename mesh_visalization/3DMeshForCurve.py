import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 读取txt文件并生成点云
def load_point_cloud_from_txt(txt_path):
    data = np.loadtxt(txt_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])  # 前三列作为点的坐标

    # 使用速度信息对点云进行颜色编码
    speeds = data[:, 3]
    norm_speeds = (speeds - speeds.min()) / (speeds.max() - speeds.min())  # 归一化速度
    cmap = plt.get_cmap("jet")  # 使用jet颜色映射
    colors = cmap(norm_speeds)[:, :3]  # 提取RGB值
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

# 路径到上传的txt文件
txt_path = r'F:\袁心辰研究生资料\4DPH_anaylsis\all_mPAP_4dnii_data171_258\Subject_171\rv_es_curvature.txt'
txt_path2 = r'F:\袁心辰研究生资料\4DPH_anaylsis\all_mPAP_4dnii_data171_258\Subject_172\rv_es_curvature.txt'

# 加载点云数据
point_cloud = load_point_cloud_from_txt(txt_path)
point_cloud2 = load_point_cloud_from_txt(txt_path2)

# 可视化点云
o3d.visualization.draw_geometries([point_cloud])
o3d.visualization.draw_geometries([point_cloud2])
