import open3d as o3d
import os
import numpy as np

# 定义文件夹路径
folder_path = r"D:\A\Curve0_1"  # 替换为实际文件夹路径

# 获取文件夹中所有 .ply 文件
ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]

# 创建一个可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window()

# 获取渲染选项并设置背景颜色为黑色
render_option = vis.get_render_option()
render_option.background_color = [0, 0, 0]  # 设置黑色背景 (R, G, B)

# 增加点的大小
render_option.point_size = 10  # 设置点大小，默认值通常为 1，你可以根据需要调整

# 增强颜色亮度的因子
color_brightness_factor = 3.5  # 增加颜色亮度的因子

# 遍历所有 .ply 文件并可视化
for ply_file in ply_files:
    # 构建文件路径
    ply_path = os.path.join(folder_path, ply_file)

    # 打印路径，确认文件是否存在
    print(f"Reading file: {ply_path}")

    try:
        # 读取点云文件
        pcd = o3d.io.read_point_cloud(ply_path)

        # 确保文件读取成功
        if not pcd.has_points():
            print(f"Warning: {ply_file} has no points or is not a valid point cloud file.")
            continue

        # 强制设置为淡蓝色 (R=0.5, G=0.5, B=1)
        print(f"{ply_file} contains color information. Forcing light blue color.")

        # 获取点云中的点数
        num_points = np.asarray(pcd.points).shape[0]

        # 创建一个淡蓝色的颜色数组
        light_blue_color = np.zeros((num_points, 3))
        light_blue_color[:, 2] = 1  # 蓝色分量为 1
        light_blue_color[:, 0] = 1  # 红色分量为 0.5
        light_blue_color[:, 1] = 1  # 绿色分量为 0.5

        # 增加亮度
        light_blue_color = np.clip(light_blue_color * color_brightness_factor, 0, 1)

        # 设置点云颜色
        pcd.colors = o3d.utility.Vector3dVector(light_blue_color)

        # 将点云添加到可视化窗口
        vis.add_geometry(pcd)

        # 进入交互式视角调整
        print(f"Please adjust the view and close the window when done.")
        vis.run()  # 进入交互模式，允许你手动调整视角

        # 捕捉屏幕图像并保存
        image_path = os.path.join(folder_path, f"{os.path.splitext(ply_file)[0]}_viewnorm.png")
        vis.capture_screen_image(image_path)
        print(f"Saved image to {image_path}")

        # 清理当前点云数据，避免添加多个点云时发生内存问题
        vis.clear_geometries()

    except Exception as e:
        print(f"Error reading {ply_file}: {e}")

# 关闭可视化窗口
vis.destroy_window()
