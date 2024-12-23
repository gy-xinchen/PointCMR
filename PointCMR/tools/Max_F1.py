# -*- coding: utf-8 -*-
# @Time    : 2024/12/4 16:44
# @Author  : yuan
# @File    : Max_F1.py
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

for i in range(5):
    # 读取文件数据
    file_path = (r'F:\袁心辰研究生资料\肺动脉高压点云回归可行性研究\miccai实验设计\2_实验数据记录\其他网络对比实验\Texture\prob\all_targets_probs{}.txt'.format(i))  # 替换为你的文件路径
    data = pd.read_csv(file_path)

    # 提取目标值和概率值
    y_true = data['Target']
    y_prob = data[' Probaility']

    # 将概率值转换为预测类别，采用0.5作为阈值
    y_pred = (y_prob >= 0.5).astype(int)

    # 计算ACC和F1值
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 输出结果
    print(f'Fold: {i}')
    print(f'Accuracy: {acc}')
    print(f'F1 Score: {f1}')

    # 为了计算最大化F1值时的ACC与F1，我们可以用不同的阈值尝试
    thresholds = np.linspace(0, 1, 101)
    best_f1 = 0
    best_acc = 0
    best_threshold = 0

    for threshold in thresholds:
        y_pred_threshold = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred_threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_acc = accuracy_score(y_true, y_pred_threshold)
            best_threshold = threshold

    # 输出最佳阈值对应的ACC和F1
    print(f'Best F1 Score: {best_f1} at threshold {best_threshold}')
    print(f'Best Accuracy: {best_acc} at threshold {best_threshold}')
