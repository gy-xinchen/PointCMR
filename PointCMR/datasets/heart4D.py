import os
import sys
import numpy as np
from torch.utils.data import Dataset
import re
import random

def extract_key(filename):
    match = re.search(r'Subject_(\d+)_(\d+)\.npz$', filename)
    if match:
        return int(match.group(1))
    return float('inf')

class Heart4D(Dataset):
    def __init__(self, root, frames_per_clip=1, frame_interval=1, num_points=2048, train=True, fold=0, num_folds=5):
        super(Heart4D, self).__init__()

        self.videos = []
        self.labels = []
        # self.position_encoding = []
        self.subject_index_map = []
        self.index_map = []

        index = 0
        sorted_files = sorted(os.listdir(root), key=extract_key)

        # 总视频数量
        num_videos = len(sorted_files)
        # 计算每折的视频数量
        fold_size = num_videos // num_folds

        # 确定当前折的训练和测试集索引范围
        test_start = fold * fold_size
        test_end = test_start + fold_size

        # 分配数据到训练集和测试集
        for i, video_name in enumerate(sorted_files):
            video = np.load(os.path.join(root, video_name))["motion"]
            # position_encoding
            # pe = np.load(os.path.join(root, video_name))["position_encoding"]
            label = int(video_name.split('_')[2].split(".")[0]) # 从文件名中提取标签
            subject_index = int(video_name.split('_')[1])
            if test_start <= i < test_end:
                # 当前视频是测试集的一部分
                if not train:
                    self.videos.append(video)
                    self.labels.append(label)
                    self.subject_index_map.append(subject_index)
                    # self.position_encoding.append(pe)

                    nframes = video.shape[0]
                    for t in range(0, nframes - frame_interval * (frames_per_clip - 1)):
                        self.index_map.append((index, t))
                    index += 1
            else:
                # 当前视频是训练集的一部分
                if train:
                    self.videos.append(video)
                    self.labels.append(label)
                    self.subject_index_map.append(subject_index)
                    # self.position_encoding.append(pe)

                    nframes = video.shape[0]
                    for t in range(0, nframes - frame_interval * (frames_per_clip - 1)):
                        self.index_map.append((index, t))
                    index += 1

        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]
        subject_index = self.subject_index_map[idx]
        # pe = self.position_encoding[index]

        clip = [video[t + i * self.frame_interval] for i in range(self.frames_per_clip)]

        # # 遍历每一帧并应用过滤条件，仅保留符合 x 轴条件的点
        # filtered_data = []
        # for frame in clip:
        #     minx, maxx = (frame[:, 0].min()+frame[:, 0].max())/2, frame[:, 0].max()
        #     mask = (frame[:, 0] >= minx) & (frame[:, 0] <= maxx)  # 选择 x 值在 [-40, -20] 范围内的点
        #     filtered_frame = frame[mask]  # 应用掩码
        #     filtered_data.append(filtered_frame)

        # # 转换成 numpy 数组，保证 shape 的一致性
        # filtered_clip = np.array(filtered_data, dtype=object)  # 使用 object 数组以适应每帧可能不同的点数

        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip) # 固定一个片段2048个点云数据,这里使用随机采样策略，我觉得需要更换采样策略

        # clip = np.concatenate((clip[15:20,:,:], clip[0:5,:,:]), axis=0)

        # if self.train:
        #     for i in range(clip.shape[0]):  # 遍历 batch 中的每个样本
        #         # 随机打乱时间维度的索引
        #         time_indices = np.random.permutation(clip.shape[1])  # 获取随机的时间维度索引顺序
        #         clip[i] = clip[i][time_indices]  # 按照随机的时间维度索引重新排列



        return clip.astype(np.float32), label, index, subject_index

class Heart4D_Multi(Dataset):
    def __init__(self, root1, root2, frames_per_clip=1, frame_interval=1, num_points=2048, train=True, fold=0, num_folds=5):
        super(Heart4D_Multi, self).__init__()

        self.videos = []
        self.labels = []
        self.phases = []
        # self.position_encoding = []
        self.subject_index_map = []
        self.index_map = []

        index = 0
        sorted_files = sorted(os.listdir(root1), key=extract_key)


        # 总视频数量
        num_videos = len(sorted_files)
        # 计算每折的视频数量
        fold_size = num_videos // num_folds

        # 确定当前折的训练和测试集索引范围
        test_start = fold * fold_size
        test_end = test_start + fold_size

        # 分配数据到训练集和测试集
        for i, video_name in enumerate(sorted_files):
            video = np.load(os.path.join(root1, video_name))["motion"]
            phase = np.load(os.path.join(root2, video_name))["point"]
            # position_encoding
            # pe = np.load(os.path.join(root, video_name))["position_encoding"]
            label = int(video_name.split('_')[2].split(".")[0]) # 从文件名中提取标签
            subject_index = int(video_name.split('_')[1])
            if test_start <= i < test_end:
                # 当前视频是测试集的一部分
                if not train:
                    self.videos.append(video)
                    self.labels.append(label)
                    self.phases.append(phase)
                    self.subject_index_map.append(subject_index)
                    # self.position_encoding.append(pe)

                    nframes = video.shape[0]
                    for t in range(0, nframes - frame_interval * (frames_per_clip - 1)):
                        self.index_map.append((index, t))
                    index += 1
            else:
                # 当前视频是训练集的一部分
                if train:
                    self.videos.append(video)
                    self.labels.append(label)
                    self.phases.append(phase)
                    self.subject_index_map.append(subject_index)
                    # self.position_encoding.append(pe)

                    nframes = video.shape[0]
                    for t in range(0, nframes - frame_interval * (frames_per_clip - 1)):
                        self.index_map.append((index, t))
                    index += 1

        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        phase = self.phases[index]
        label = self.labels[index]
        subject_index = self.subject_index_map[idx]
        # pe = self.position_encoding[index]

        clip = [video[t + i * self.frame_interval] for i in range(self.frames_per_clip)]

        # # 遍历每一帧并应用过滤条件，仅保留符合 x 轴条件的点
        # filtered_data = []
        # for frame in clip:
        #     minx, maxx = (frame[:, 0].min()+frame[:, 0].max())/2, frame[:, 0].max()
        #     mask = (frame[:, 0] >= minx) & (frame[:, 0] <= maxx)  # 选择 x 值在 [-40, -20] 范围内的点
        #     filtered_frame = frame[mask]  # 应用掩码
        #     filtered_data.append(filtered_frame)

        # # 转换成 numpy 数组，保证 shape 的一致性
        # filtered_clip = np.array(filtered_data, dtype=object)  # 使用 object 数组以适应每帧可能不同的点数

        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip) # 固定一个片段2048个点云数据,这里使用随机采样策略，我觉得需要更换采样策略
        # clip = clip / 300

        num_points_in_clip = phase.shape[0]
        if num_points_in_clip > self.num_points:
            r = np.random.choice(num_points_in_clip, size=self.num_points, replace=False)
            phase = phase[r, :]  # 随机选择 num_points 个点
        phase = phase / 300
        # clip = np.concatenate((clip[15:20,:,:], clip[0:5,:,:]), axis=0)

        # if self.train:
        #     for i in range(clip.shape[0]):  # 遍历 batch 中的每个样本
        #         # 随机打乱时间维度的索引
        #         time_indices = np.random.permutation(clip.shape[1])  # 获取随机的时间维度索引顺序
        #         clip[i] = clip[i][time_indices]  # 按照随机的时间维度索引重新排列



        return clip.astype(np.float32), label, phase, index, subject_index



if __name__ == '__main__':
    dataset = Heart4D(root=r'/root/autodl-tmp/data/4D_motion_classification35_npz', frames_per_clip=16)
    clip, label, video_idx = dataset[0]
    print(clip)
    print(label)
    print(video_idx)
    print(dataset.num_classes)
