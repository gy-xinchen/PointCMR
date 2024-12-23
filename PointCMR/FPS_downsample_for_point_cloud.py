# -*- coding: utf-8 -*-
# @Time    : 2024/8/19 14:08
# @Author  : yuan
# @File    : FPS_downsample_for_point_cloud.py
import os.path

import numpy as np

def farthest_point_sampling(points, num_samples):
    """
    使用最远点采样策略从点云中采样指定数量的点。

    参数:
        points (numpy.ndarray): 输入的点云数据，形状为 (N, D)，其中 N 是点的数量，D 是每个点的维度。
        num_samples (int): 需要采样的点的数量。

    返回:
        numpy.ndarray: 采样后的点云数据，形状为 (num_samples, D)。
    """
    N, D = points.shape
    if N <= num_samples:
        return points, np.arange(N)

    # 初始化距离数组
    distances = np.full(N, np.inf)
    # 随机选择第一个点
    farthest_pts = np.zeros((num_samples,), dtype=int)
    farthest_pts[0] = np.random.randint(0, N)
    for i in range(1, num_samples):
        # 更新距离
        dist = np.sum((points - points[farthest_pts[i - 1], :]) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest_pts[i] = np.argmax(distances)

    return  farthest_pts

dir_path = r"/root/autodl-tmp/PST-Transformer/data/4D_motion_classification35_position_encoding_npy"
output_path = r"/root/autodl-tmp/PST-Transformer/data/4D_motion_classifcation35_npy_2048point"
files = os.listdir(dir_path)


for i in range(len(files)):
    npz_file = np.load(os.path.join(dir_path, files[i]))["motion"]
    clips = []
    for j in range(npz_file.shape[0]):
        r = farthest_point_sampling(npz_file[j,:], 2048)
        clip = npz_file[j, r, :]
        clips.append(clip)
    clips = np.array(clips)
    np.savez(os.path.join(output_path, files[i]), clips=clips)
    print(f"{files[i]} done")