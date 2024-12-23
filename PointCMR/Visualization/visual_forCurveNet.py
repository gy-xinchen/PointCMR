import torch
import os
import time
import datetime
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch import nn
import models.sequence_classification as Models
from fusion_PST_CurveNet.CurveNet_model.curvenet_cls import CurveNetMINE
from datasets.heart4D import Heart4D_Multi
import pandas as pd
from sklearn.preprocessing import StandardScaler
import utils
import torch.nn.functional as F
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler


def feature_propagation(low_res_xyzs, high_res_xyzs, low_res_features):
    """
    使用简单的距离加权插值方法将低分辨率点云的特征传播到高分辨率点云。

    参数:
    - low_res_xyzs: 低分辨率点云的坐标，形状为 [b, n_low, 3]
    - high_res_xyzs: 高分辨率点云的坐标，形状为 [b, n_high, 3]
    - low_res_features: 低分辨率点云的特征，形状为 [b, n_low, d]，其中 d 是特征维度

    返回:
    - high_res_features: 高分辨率点云的特征，形状为 [b, n_high, d]
    """
    b, n_low, _ = low_res_xyzs.size()
    _, n_high, _ = high_res_xyzs.size()

    # 计算低分辨率点云与高分辨率点云的欧几里得距离
    low_res_xyzs_expanded = low_res_xyzs.unsqueeze(2)  # 形状 [b, n_low, 3, 1]
    high_res_xyzs_expanded = high_res_xyzs.unsqueeze(1)  # 形状 [b, 1, n_high, 3]
    distances = torch.norm(low_res_xyzs_expanded - high_res_xyzs_expanded, dim=-1)  # 形状 [b, n_low, n_high]

    # 对距离进行归一化处理，避免数值问题
    distances = distances + 1e-8  # 加一个小的常数以避免除零错误
    weights = 1.0 / distances  # 使用反距离作为权重

    # 对权重进行归一化
    weights_sum = weights.sum(dim=1, keepdim=True)
    normalized_weights = weights / weights_sum  # 归一化后的权重

    # 使用加权平均值进行特征传播
    high_res_features = torch.matmul(normalized_weights.permute(0, 2, 1), low_res_features)  # [b, n_high, d]

    return high_res_features
def feature_propagation_nobatch(low_res_xyzs, high_res_xyzs, low_res_features):
    """
    使用简单的距离加权插值方法将低分辨率点云的特征传播到高分辨率点云。

    参数:
    - low_res_xyzs: 低分辨率点云的坐标，形状为 [n_low, 6]
    - high_res_xyzs: 高分辨率点云的坐标，形状为 [n_high, 6]
    - low_res_features: 低分辨率点云的特征，形状为 [n_low, d]，其中 d 是特征维度

    返回:
    - high_res_features: 高分辨率点云的特征，形状为 [n_high, d]
    """
    n_low, _ = low_res_xyzs.size()
    n_high, _ = high_res_xyzs.size()

    # 计算低分辨率点云与高分辨率点云的欧几里得距离
    low_res_xyzs_expanded = low_res_xyzs.unsqueeze(1)  # 形状 [n_low, 1, 3]
    high_res_xyzs_expanded = high_res_xyzs.unsqueeze(0)  # 形状 [1, n_high, 3]
    distances = torch.norm(low_res_xyzs_expanded - high_res_xyzs_expanded, dim=-1)  # 形状 [n_low, n_high]

    # 对距离进行归一化处理，避免数值问题
    distances = distances + 1e-8  # 加一个小的常数以避免除零错误
    weights = 1.0 / distances  # 使用反距离作为权重

    # 对权重进行归一化
    weights_sum = weights.sum(dim=0, keepdim=True)  # [1, n_high]
    normalized_weights = weights / weights_sum  # 归一化后的权重

    # 使用加权平均值进行特征传播
    high_res_features = torch.matmul(normalized_weights.T, low_res_features)  # [n_high, d]

    return high_res_features
def get_text_features(index_list, columns):
    data = pd.read_csv(args.txt_data, index_col=0)  # 假设第0列是索引列
    # 根据索引列表和指定列获取数据
    index_list = index_list.tolist()
    # 根据索引列表和指定列获取数据
    selected_data = data.loc[index_list, columns]  # 直接选择多个索引和列
    return selected_data.values  # 返回为一个 NumPy 数组

class MINE(nn.Module):
    def __init__(self, feature_dim, hidden_dim_mine=128):
        super(MINE, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim_mine),
            nn.ReLU(),
            nn.Linear(hidden_dim_mine, 1)
        )

    def forward(self, f_A, f_B):
        # 拼接特征
        joint = torch.cat([f_A, f_B], dim=1)
        # 计算联合分布
        output = self.fc(joint)
        return output


def test(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("torch version: ", torch.__version__)

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 加载测试数据
    print("Loading test data")
    dataset_test = Heart4D_Multi(
        root1=args.data_path1,
        root2=args.data_path2,
        frames_per_clip=args.clip_len,
        frame_interval=args.frame_interval,
        num_points=args.num_points,
        train=False,  # 只加载测试数据
        fold=0,  # 测试阶段不需要fold
        num_folds=1
    )

    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    # 创建模型
    print("Creating model")
    Model = getattr(Models, args.model)
    model = Model(
        radius=args.radius,
        nsamples=args.nsamples,
        spatial_stride=args.spatial_stride,
        temporal_kernel_size=args.temporal_kernel_size,
        temporal_stride=args.temporal_stride,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        dropout1=args.dropout1,
        mlp_dim=args.mlp_dim,
        num_classes=2,
        dropout2=args.dropout2,
    )

    # 创建CurveNet的3D point cloud模型
    Model_3Dphase = CurveNetMINE(num_classes=2).to(device)
    mine_net = MINE(128).to(device)

    # 加载权重
    if args.resume:
        print(f"Loading model checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        Model_3Dphase.load_state_dict(checkpoint['model_3Dphase'])
        mine_net.load_state_dict(checkpoint['mine_net'])

    model.to(device)
    Model_3Dphase.to(device)
    mine_net.to(device)

    # 开始测试
    print("Starting evaluation")
    model.eval()  # 切换到评估模式
    Model_3Dphase.eval()
    mine_net.eval()

    auc_value = 0
    auc_per_fold = []
    predictions = []
    ground_truth = []

    with torch.no_grad():  # 在测试过程中不计算梯度
        for batch_idx, (clip, target, phase, video_idx, subject_index) in enumerate(data_loader_test):
            columns = ["室间隔夹角", "心室质量指数"]
            text_features = get_text_features(subject_index, columns)[:, 0:2]
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(text_features)

            clip, target, phase = clip.to(device), target.to(device), phase.to(device)
            phase = phase.permute(0, 2, 1).float()

            # 获取模型输出
            output, _, attn_score, xyzs = model(clip, normalized_features)
            # last_attn_score = attn_score[-1].squeeze(dim=-1) # 最后一层权重 [batch, head, m, m]
            # one_head_last_attn_score = last_attn_score[:, 0, :, :].squeeze(dim=1) # 获得一个头的注意力
            # xyzs_map = xyzs.view(2, 320, 3) # 重塑位置的形状
            # weighted_xyzs_map = torch.matmul(one_head_last_attn_score, xyzs_map) # 将位置与注意力权重加权
            # reshaped_weighted_xyzs_map = weighted_xyzs_map.view(2, 10, 32, 3).cpu().numpy() # 这里得到的是重塑后的注意力加权矩阵

            # 这里需要将初始化conv高纬映射还原 [b, 10, 128, 3] --> [b, 20, 2048, 3]
            output_phase, _, attn = Model_3Dphase(phase)
            phase_xyz_pos = attn[0] # [batch, channel, points]
            phase_feature_map = attn[1] # [batch, channel, point]


            for sample in range(args.batch_size):
                phase_feature_map = F.adaptive_max_pool2d(phase_feature_map, 64) # 将其特征图平均池化512--64
                weighted_xyzs_map = torch.matmul(phase_xyz_pos[sample, :, :], phase_feature_map[sample, :, :].T) # [6,64]*[64,64]==[6,64]

                high_res_features = feature_propagation_nobatch(
                    phase_xyz_pos[sample, :, :].transpose(1,0),  # 根据当前帧选择 xyzs 数据 [64, 6]
                    phase[sample, :, :].transpose(1,0),  # 根据当前帧选择 clip 数据 [1024,6]
                    weighted_xyzs_map.transpose(1,0)  # 根据当前帧选择加权 xyzs [64,6]
                )

                high_res_features = high_res_features.transpose(1,0)

                # high_res_features是将其恢复后的注意力加权矩阵 形状为 [6, 1024]
                normalized_high_res_features = (high_res_features - high_res_features.min()) / (
                        high_res_features.max() - high_res_features.min())

                weighted_avg = normalized_high_res_features.cpu().numpy()
                weighted_avg = np.mean(weighted_avg, axis=0)
                weighted_avg = (weighted_avg - weighted_avg.min()) / (weighted_avg.max() - weighted_avg.min())

                # 提取当前时间步的点云数据 (形状为 [1024, 6])
                points = phase[sample, :3, :].cpu().numpy() .transpose() # 使用 frame 获取当前时间步的点云数据

                # 将权重映射到颜色
                colors = plt.cm.jet(weighted_avg)[:, :3]

                # 创建 Open3D 点云对象
                pcd = o3d.geometry.PointCloud()

                # 将点云数据传入 Open3D 点云对象
                pcd.points = o3d.utility.Vector3dVector(points)

                # 将颜色信息传入 Open3D 点云对象
                pcd.colors = o3d.utility.Vector3dVector(colors)

                # 保存路径
                save_dir = f"/root/autodl-tmp/PST-copy/gif/{sample}_{batch_idx}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 保存点云为PLY文件
                ply_filename = f"{save_dir}/point_cloud_CuveNet.ply"
                o3d.io.write_point_cloud(ply_filename, pcd)


            prob = F.softmax(input=output, dim=1)
            prob_phase = F.softmax(input=output_phase, dim=1)

            # 获取得到最大类的概率值
            max_prob_values, _ = torch.max(prob, dim=1)
            max_prob_values_phase, _ = torch.max(prob_phase, dim=1)

            result_prob = (max_prob_values + max_prob_values_phase) / 2
            print(f"result_prob {result_prob} ## target {target}")


            # 存储预测和真实标签
            predictions.append(result_prob.cpu().numpy())
            ground_truth.append(target.cpu().numpy())

    # 保存预测结果
    if args.output_dir:
        output_file = os.path.join(args.output_dir, "test_predictions.npy")
        np.save(output_file, np.concatenate(predictions, axis=0))
        print(f"Predictions saved to {output_file}")

    return auc_value


def calculate_auc(output, target):
    from sklearn.metrics import roc_auc_score
    # 将模型输出转为概率
    output = torch.softmax(output, dim=1)
    output = output[:, 1]  # 选择目标类的概率（假设二分类问题）
    return roc_auc_score(target.cpu().numpy(), output.cpu().numpy())


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Test PST-Transformer Model')

    # 数据和模型相关参数
    parser.add_argument('--data-path1', default=r'/root/autodl-tmp/PST-copy/data/3_可视化/4D', type=str, help='dataset')
    parser.add_argument('--data-path2', default=r'/root/autodl-tmp/PST-copy/data/3_可视化/3D', type=str, help='dataset')
    parser.add_argument('--txt_data', default='/root/autodl-tmp/PST-copy/data/txt_dataset_fold0_test.csv', type=str, help='txt-dataset')

    # 输出和权重恢复
    parser.add_argument('--output-dir', default='/root/autodl-tmp/PST-copy/predict_npy', type=str, help='path where to save predictions')
    parser.add_argument('--resume', default='/root/autodl-tmp/dataword/PST_CurveNet1024_txt_MINE_ALL/weight/checkpoint_0.pth', type=str, help='path to checkpoint for resume')

    parser.add_argument('--model', default='PSTTransformerFusionTxtMINE', type=str, help='model')
    parser.add_argument('--seed', default=42, type=int)

    # 输入参数
    parser.add_argument('--clip-len', default=20, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--frame-interval', default=1, type=int, metavar='N', help='interval of sampled frames')
    parser.add_argument('--num-points', default=1024, type=int, metavar='N', help='number of points per frame')

    # P4D 和 Transformer参数
    parser.add_argument('--radius', default=0.3, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=32, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal-stride', default=2, type=int, help='temporal stride')

    parser.add_argument('--dim', default=80, type=int, help='transformer dim')
    parser.add_argument('--depth', default=5, type=int, help='transformer depth')
    parser.add_argument('--heads', default=2, type=int, help='transformer head')
    parser.add_argument('--dim-head', default=40, type=int, help='transformer dim for each head')
    parser.add_argument('--mlp-dim', default=160, type=int, help='transformer mlp dim')
    parser.add_argument('--dropout1', default=0.0, type=float, help='transformer dropout')

    parser.add_argument('--dropout2', default=0.0, type=float, help='classifier dropout')

    # 测试相关参数
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    test_auc = test(args)
    print(f"Final Test AUC: {test_auc:.4f}")
