import numpy as np
import open3d as o3d

# 加载点云数据
data = np.load(r'F:\袁心辰研究生资料\code\PST-Transformer-main\data\4D_motion_classification35_npz\Subject_10_1.npz')["motion"]
print(f"point_clouds的形状: {data.shape}")

# 创建Open3D点云对象的辅助函数
def create_point_cloud_from_numpy(points):
    """从NumPy数组创建Open3D点云对象"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# 定义曲率计算函数
def compute_curvature(points):
    """计算点云的曲率"""
    pcd = create_point_cloud_from_numpy(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = []
    for i in range(len(points)):
        # 使用 KDTree 查找近邻点
        [k, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], knn=10)
        normals = np.asarray(pcd.normals)[idx, :]
        curvature = np.std(normals, axis=0).sum()
        curvatures.append(curvature)
    return np.array(curvatures)

# 初始化存储曲率最大的点的列表
filtered_data = []

# 逐帧处理点云数据，选取曲率最大的2048个点
for frame in data:
    curvatures = compute_curvature(frame)
    # 获取曲率最高的2048个点的索引
    if len(curvatures) >= 2048:
        indices = np.argsort(-curvatures)[:2048]
        filtered_frame = frame[indices]
    else:
        filtered_frame = frame  # 若点数不足2048，则保留所有点
    filtered_data.append(filtered_frame)

# 转换成 numpy 数组，保证 shape 的一致性
filtered_data = np.array(filtered_data, dtype=object)  # 使用 object 数组以适应每帧可能不同的点数

# 初始化可视化窗口
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# 初始化第一个帧的点云对象
pcd = create_point_cloud_from_numpy(filtered_data[0])
vis.add_geometry(pcd)

# 更新点云数据的函数
def update_point_cloud(frame):
    """更新当前帧的点云数据"""
    pcd.points = o3d.utility.Vector3dVector(filtered_data[frame])
    vis.update_geometry(pcd)

# 设置初始帧索引
frame_index = [0]  # 使用列表封装frame_index以便在函数内修改

# 定义键盘回调函数，按空格键切换到下一帧
def next_frame(vis):
    """切换到下一帧"""
    frame_index[0] = (frame_index[0] + 1) % len(data)  # 环绕至第一个帧
    update_point_cloud(frame_index[0])
    return False  # 返回False使得窗口保持开启

# 注册按键回调函数，按下空格键后切换到下一帧
vis.register_key_callback(ord(" "), next_frame)

# 开始可视化窗口，允许用户手动调整视角并按空格切换帧
vis.run()
vis.destroy_window()
