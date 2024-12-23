# -*- coding: utf-8 -*-
# @Time    : 2024/12/2 15:04
# @Author  : yuan
# @File    : train-heart4D-fusion.py
from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import utils
import pandas as pd
from scheduler import WarmupMultiStepLR
from sklearn.metrics import auc as sklearn_auc, roc_curve, accuracy_score, f1_score, matthews_corrcoef
from datasets.heart4D import Heart4D_Multi
import models.sequence_classification as Models
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from thop import profile
from torch.optim.lr_scheduler import CosineAnnealingLR
from fusion_PST_CurveNet.CurveNet_model.curvenet_cls import CurveNet

class TextPointCloudFusionModel(nn.Module):
    def __init__(self, base_model, text_feature_dim, output_dim):
        super(TextPointCloudFusionModel, self).__init__()
        self.base_model = base_model  # Transformer 基础模型
        self.fc_text = nn.Sequential(
            nn.Linear(text_feature_dim, 20),  # 升维
            nn.ReLU(),
            nn.Linear(20, 11))  # 降维
        self.fc_fusion = nn.Sequential(
            nn.Linear(1024 + text_feature_dim, 512),  # 假设模型输出为1024维
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.softmax = nn.Softmax(dim=1)  # 对每个样本应用 Softmax，dim=1 表示按列计算

    def forward(self, clip, text_features):
        # 获取Transformer模型的输出
        point_features = self.base_model(clip)  # 假设输出为 (batch_size, 1024)
        # 将文本特征转换为张量并移到同一个设备
        text_features = torch.tensor(text_features, dtype=torch.float32, device=point_features.device)
        # 对text_feature进行均值为0方差为1的归一化
        text_features = (text_features - text_features.mean(dim=0, keepdim=True)) / (
                    text_features.std(dim=0, keepdim=True) + 1e-6)
        text_features = self.fc_text(text_features)

        # 文本特征均值为0方差为1归一化
        text_features = (text_features - text_features.mean(dim=0)) / text_features.std(dim=0)
        # 图像特征均值为0方差为1归一化
        point_features = (point_features - point_features.mean(dim=0)) / point_features.std(dim=0)

        # 将点云特征与文本特征拼接
        combined_features = torch.cat((point_features, text_features), dim=1)
        # 融合后的特征通过全连接层
        output = self.fc_fusion(combined_features)
        output = self.softmax(output)  # 计算每个样本的分类概率
        return output

def get_text_features(index_list, columns):
    data = pd.read_csv(args.txt_data, index_col=0)  # 假设第0列是索引列
    # 根据索引列表和指定列获取数据
    index_list = index_list.tolist()
    # 根据索引列表和指定列获取数据
    selected_data = data.loc[index_list, columns]  # 直接选择多个索引和列
    return selected_data.values  # 返回为一个 NumPy 数组

def calculate_auc_ci(label_list, predict_list):
    label_list = np.array(label_list)
    predict_list = np.array(predict_list)

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_auc_value = []
    np.random.seed(rng_seed)
    for i in range(n_bootstraps):
        indices = np.random.randint(0, len(predict_list), len(predict_list))
        fpr, tpr, thresholds = roc_curve(label_list[indices], predict_list[indices])
        bootstrapped_auc_value.append(sklearn_auc(fpr, tpr))
    bootstrapped_auc = np.array(bootstrapped_auc_value)
    lower_bound = np.percentile(bootstrapped_auc, 2.5)
    upper_bound = np.percentile(bootstrapped_auc, 97.5)

    return bootstrapped_auc, lower_bound, upper_bound

def label_smoothing(target, num_classes, smoothing=0.1):
    with torch.no_grad():
        # 将目标转换为 one-hot 编码
        target_one_hot = F.one_hot(target, num_classes).float()
        # 应用标签平滑
        smoothed_target = target_one_hot * (1.0 - smoothing) + (smoothing / num_classes)
    return smoothed_target

def train_one_epoch(model, Model_3Dphase, criterion, criterion_phase, optimizer, optimizer_phase, lr_scheduler, scheduler_phase, data_loader, device, epoch, print_freq):
    model.train()
    Model_3Dphase.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    epoch_loss = 0

    for clip, target, phase, index, _ in metric_logger.log_every(data_loader, print_freq, header):
        # columns = ['性别', '年龄', "体重", "身高", "体表面积", "室间隔夹角", "心室质量指数", "RVEDVi", "RVESVi", "RVSVi", "RVEFi"]
        # text_features = get_text_features(index, columns)
        start_time = time.time()
        clip, target, phase = clip.to(device), target.to(device), phase.to(device)
        phase = phase.permute(0, 2, 1).float()
        output = model(clip)
        output_phase = Model_3Dphase(phase)

        # flops, params = profile(model, (clip,))
        # print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))

        # 计算标签平滑后的目标
        num_classes = output.size(1)  # 假设 output 的第二个维度是类别数
        smoothed_target = label_smoothing(target, num_classes=num_classes, smoothing=args.label_smoothing)

        # 计算损失，使用平滑后的目标
        loss = criterion(output, smoothed_target)
        loss_phase = criterion_phase(output_phase, smoothed_target)
        all_losses = loss + loss_phase

        epoch_loss += all_losses.item()

        optimizer.zero_grad()
        optimizer_phase.zero_grad()
        loss.backward()
        loss_phase.backward()
        optimizer.step()
        optimizer_phase.step()

        acc1, acc2 = utils.accuracy(output, target, topk=(1, 2))
        acc1_phase, acc2_phase = utils.accuracy(output_phase, target, topk=(1, 2))

        batch_size = clip.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc1_phase'].update(acc1_phase.item(), n=batch_size)

        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()
        scheduler_phase.step()
        sys.stdout.flush()

    return epoch_loss / len(data_loader)

def evaluate(model, Model_3Dphase, criterion, criterion_phase, data_loader, device, best_auc):
    model.eval()
    Model_3Dphase.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    epoch_loss = 0
    video_prob = {}
    phase_prob = {}
    video_label = {}
    max_prob_label = {}
    max_prob_label_phase = {}
    all_probs = []  # to store all probabilities
    video_probs, phase_probs  = [], []
    all_targets = []  # to store all targets
    with torch.no_grad():
        for clip, target, phase, video_idx, _ in metric_logger.log_every(data_loader, 100, header):
            # columns = ['性别', '年龄', "体重", "身高", "体表面积", "室间隔夹角", "心室质量指数", "RVEDVi", "RVESVi",
            #            "RVSVi", "RVEFi"]
            # text_features = get_text_features(video_idx, columns)

            clip = clip.to(device, non_blocking=True)
            phase = phase.to(device, non_blocking=True)
            phase = phase.permute(0, 2, 1).float()
            target = target.to(device, non_blocking=True)
            output = model(clip)
            output_phase = Model_3Dphase(phase)
            loss = criterion(output, target)
            loss_phase = criterion_phase(output_phase, target)
            all_losses = loss + loss_phase
            epoch_loss += all_losses.item()

            # need change
            # 计算得到4D和3D的预测得分
            acc1, acc2 = utils.accuracy(output, target, topk=(1, 2))
            acc1_phase, acc2_phase = utils.accuracy(output_phase, target, topk=(1, 2))

            # 对预测得分进行归一化
            prob = F.softmax(input=output, dim=1)
            prob_phase = F.softmax(input=output_phase, dim=1)

            # 获取得到最大类的概率值
            max_prob_values, _ = torch.max(prob, dim=1)
            max_prob_values_phase, _ = torch.max(prob_phase, dim=1)


            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = clip.shape[0]
            target = target.cpu().numpy()
            video_idx = video_idx.cpu().numpy()
            prob = prob.cpu().numpy()
            prob_phase = prob_phase.cpu().numpy()
            max_prob_values = max_prob_values.cpu().numpy()
            max_prob_values_phase = max_prob_values_phase.cpu().numpy()

            for i in range(0, batch_size):
                idx = video_idx[i]
                if idx in video_prob:
                    video_prob[idx] += prob[i]
                    phase_prob[idx] += prob_phase[i]
                else:
                    video_prob[idx] = prob[i]
                    phase_prob[idx] = prob_phase[i]
                    video_label[idx] = target[i]
                    max_prob_label[idx] = max_prob_values[i]
                    max_prob_label_phase[idx] = max_prob_values_phase[i]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc1_phase'].update(acc1_phase.item(), n=batch_size)

            video_probs.extend(prob[:, 1])  # assuming the second column is for class 1
            phase_probs.extend(prob_phase[:, 1])
            all_targets.extend(target)

            all_probs = [(v + p) / 2 for v, p in zip(video_probs, phase_probs)]

        auc_value_video = roc_auc_score(all_targets, video_probs)
        auc_value_phase = roc_auc_score(all_targets, phase_probs)
        auc_value_all = roc_auc_score(all_targets, all_probs)

        all_probs = np.array(all_probs)
        preds = (all_probs >= 0.5).astype(int)
        accuracy = accuracy_score(all_targets, preds)
        f1 = f1_score(all_targets, preds)
        MCC = matthews_corrcoef(all_targets, preds)

        if auc_value_all > best_auc:
            best_auc = auc_value_all
            bootstrapped_auc, lower_bound, upper_bound = calculate_auc_ci(all_targets, all_probs)

            with open(os.path.join(args.output_dir, "prob", f"all_targets_probs{fold}.txt"), 'w') as f:
                f.write("Target, Probability\n")
                for target, prob in zip(all_targets, all_probs):
                    f.write(f"{target}, {prob}\n")

            fpr, tpr, thersholds = roc_curve(all_targets, all_probs)

            roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thersholds})
            wb = Workbook()
            ws = wb.active
            ws.title = "ROC"
            for r in dataframe_to_rows(roc_df, index=False, header=True):
                ws.append(r)

            auc_interval_df = pd.DataFrame(
                {"AUC": [auc_value_all], "acc": [accuracy], "f1": [f1], "MCC": [MCC], "lower Bound": [lower_bound], "Upper Bound": [upper_bound]}, index=[0])
            auc_interval_df.to_csv(os.path.join(args.output_dir, "excle", f"Bound{fold}.csv"), index=False)
            wb.save(os.path.join(args.output_dir, "excle", f"{fold}.csv"))

        print(f"prob {max_prob_label.values()}")
        print(f"target {video_label.values()}")
        print(f"auc_value_video {auc_value_video}")
        print(f"auc_value_phase {auc_value_phase}")
        print(f"auc {auc_value_all}")


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Clip Acc@1 {top1.global_avg:.3f} '.format(top1=metric_logger.acc1))

    # video level prediction
    video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
    pred_correct = [video_pred[k]==video_label[k] for k in video_pred]
    total_acc = np.mean(pred_correct)

    class_count = [0] * data_loader.dataset.num_classes
    class_correct = [0] * data_loader.dataset.num_classes

    for k, v in video_pred.items():
        label = video_label[k]
        class_count[label] += 1
        class_correct[label] += (v==label)
    class_acc = [c/float(s) for c, s in zip(class_correct, class_count)]

    print(' * Video Acc@1 %f'%total_acc)
    print(' * Class Acc@1 %s'%str(class_acc))

    return auc_value_all,  epoch_loss / len(data_loader), best_auc

def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    # Data loading code
    print("Loading data")

    st = time.time()

    dataset = Heart4D_Multi(
            root1=args.data_path1,
            root2=args.data_path2,
            frames_per_clip=args.clip_len,
            frame_interval=args.frame_interval,
            num_points=args.num_points,
            train=True,
            fold=fold,
            num_folds = args.num_folds
    )

    dataset_test = Heart4D_Multi(
            root1=args.data_path1,
            root2=args.data_path2,
            frames_per_clip=args.clip_len,
            frame_interval=args.frame_interval,
            num_points=args.num_points,
            train=False,
            fold=fold,
            num_folds = args.num_folds
    )

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    # 创建PST的4D point cloud模型
    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
                  temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
                  dim=args.dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head, dropout1=args.dropout1,
                  mlp_dim=args.mlp_dim, num_classes=2, dropout2=args.dropout2, )
    # 创建CurveNet的3D point cloud模型
    Model_3Dphase = CurveNet(num_classes=2).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # 定义PST的损失
    criterion = nn.CrossEntropyLoss()
    # # 定义CurveNet损失
    criterion_phase = nn.CrossEntropyLoss()

    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_phase = torch.optim.Adam(Model_3Dphase.parameters(), lr=args.lr_phase, weight_decay=1e-4)
    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)
    scheduler_phase = CosineAnnealingLR(optimizer_phase, args.epochs, eta_min=1e-3)


    model_without_ddp = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, "png")):
        os.makedirs(os.path.join(args.output_dir, "png"))
    if not os.path.exists(os.path.join(args.output_dir, "weight")):
        os.makedirs(os.path.join(args.output_dir, "weight"))
    if not os.path.exists(os.path.join(args.output_dir, "excle")):
        os.makedirs(os.path.join(args.output_dir, "excle"))
    if not os.path.exists(os.path.join(args.output_dir, "prob")):
        os.makedirs(os.path.join(args.output_dir, "prob"))

    print("Start training")
    start_time = time.time()
    auc_value = 0
    best_auc = 0
    weight_best_auc = 0
    auc_per_fold = []
    train_losses, val_losses = [], []
    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train_one_epoch(model, Model_3Dphase, criterion, criterion_phase, optimizer, optimizer_phase, lr_scheduler, scheduler_phase, data_loader, device, epoch, args.print_freq)
        auc_data, val_loss, best_auc =  evaluate(model, Model_3Dphase, criterion, criterion_phase, data_loader_test, device=device, best_auc=best_auc)


        train_losses.append(train_loss)
        val_losses.append(val_loss)



        if auc_data > weight_best_auc:
            weight_best_auc = auc_data
            if args.output_dir:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}
                # utils.save_on_master(
                #     checkpoint,
                #     os.path.join(args.output_dir, "weight", 'model_{}.pth'.format(epoch)))
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, "weight", f'checkpoint_{fold}.pth'))
        auc_value = max(auc_value, auc_data)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.savefig(os.path.join(args.output_dir, "png", f'training_validation_fold{fold}_loss.png'), format='png', dpi=300)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("AUC {}".format(auc_value))

    auc_per_fold.append(auc_value)
    best_auc = max(auc_per_fold)

    return best_auc


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PST-Transformer Model Training')

    parser.add_argument('--data-path1', default='/root/autodl-tmp/data/correct_4D_motion_classification35_position_encoding_npz', type=str, help='dataset')
    parser.add_argument('--data-path2', default='/root/autodl-tmp/PST-copy/data/3D_phase_systole', type=str, help='dataset')
    parser.add_argument('--txt_data', default='/root/autodl-tmp/data/txt_dataset.csv', type=str, help='txt-dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='PSTTransformer', type=str, help='model')
    # input
    parser.add_argument('--clip-len', default=20, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--frame-interval', default=1, type=int, metavar='N', help='interval of sampled frames')
    parser.add_argument('--num-points', default=4096, type=int, metavar='N', help='number of points per frame')
    # P4D
    parser.add_argument('--radius', default=0.3, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=32, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal-stride', default=2, type=int, help='temporal stride')
    # transformer
    parser.add_argument('--dim', default=80, type=int, help='transformer dim')
    parser.add_argument('--depth', default=5, type=int, help='transformer depth')
    parser.add_argument('--heads', default=2, type=int, help='transformer head')
    parser.add_argument('--dim-head', default=40, type=int, help='transformer dim for each head')
    parser.add_argument('--mlp-dim', default=160, type=int, help='transformer mlp dim')
    parser.add_argument('--dropout1', default=0.0, type=float, help='transformer dropout')
    # output
    parser.add_argument('--dropout2', default=0.0, type=float, help='classifier dropout')
    # training
    parser.add_argument('-b', '--batch-size', default=14, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr_phase', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=3, type=int, help='number of warmup epochs')
    # output
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default=r'/root/autodl-tmp/dataword/PST_CurveNet4096', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    # n fold
    parser.add_argument("-num_folds", default=5, type=int, help="fold")
    # label smoothing
    parser.add_argument("-label_smoothing", default=0.0, type=int, help="label_smoothing")

    args = parser.parse_args()


    return args

if __name__ == "__main__":
    args = parse_args()
    five_fold_auc = []
    for fold in range(args.num_folds): # 0,1,2,3,4
        auc_values = main(args)
        five_fold_auc.append(auc_values)
        print("five fold auc {}".format(five_fold_auc))
