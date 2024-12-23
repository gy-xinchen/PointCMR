import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from point_4d_convolution import *
from transformer_v1 import *

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.register_buffer('pe', self._positional_encoding(max_len, embedding_dim))

    def _positional_encoding(self, max_len, embedding_dim):
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(20) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        # x: [Batch, frames, embedding_point, channels]
        seq_len = x.size(1)
        return self.pe[:, :seq_len]  # Return positional encoding for frames

class PSTTransformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 dim, depth, heads, dim_head, dropout1,                                 # transformer
                 mlp_dim, num_classes, dropout2):                                       # output
        super().__init__()
        self.PositionalEncoding = PositionalEncoding(64,10)
        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L_frame, Channel, n_point]

        # # 合并降采样
        # pe_reshaped = pe.view(pe.shape[0], pe.shape[1] // 2, 2)
        # # 取均值
        # pe_new = pe_reshaped.mean(dim=2)
        # # 生成序列变化长度
        # sequence = torch.arange(1, pe_new.shape[1] + 1).unsqueeze(0).repeat(pe_new.shape[0], 1).to(device)
        # 生成时序差异的时间特征
        # pe_sequence = pe_new + sequence
        # 扩充维度
        # pe_sequence = pe_new.to(torch.float32).unsqueeze(dim=2).unsqueeze(dim=3)




        # time_embedding = self.PositionalEncoding(features).unsqueeze(dim=2).to(device) # our_time_embedding = [batch, frame, 1, channel]
        # # conv for time
        # features += time_embedding # feature = [batch, frame, dim, channel]

        features = features.permute(0, 1, 3, 2) # [8, 10, 64, 80][batch, L_frame, n_point, Channel]

        output = self.transformer(xyzs, features)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output


class PSTTransformerFusionTxt(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 dim, depth, heads, dim_head, dropout1,                                 # transformer
                 mlp_dim, num_classes, dropout2):                                       # output
        super().__init__()
        self.PositionalEncoding = PositionalEncoding(64,10)
        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, 80),
        )

        self.mlp_fusion = nn.Sequential(
            nn.GELU(),
            nn.Linear(82, 200),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(200, num_classes),
        )

    def forward(self, input, txt_feature):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L_frame, Channel, n_point]

        # # 合并降采样
        # pe_reshaped = pe.view(pe.shape[0], pe.shape[1] // 2, 2)
        # # 取均值
        # pe_new = pe_reshaped.mean(dim=2)
        # # 生成序列变化长度
        # sequence = torch.arange(1, pe_new.shape[1] + 1).unsqueeze(0).repeat(pe_new.shape[0], 1).to(device)
        # 生成时序差异的时间特征
        # pe_sequence = pe_new + sequence
        # 扩充维度
        # pe_sequence = pe_new.to(torch.float32).unsqueeze(dim=2).unsqueeze(dim=3)




        # time_embedding = self.PositionalEncoding(features).unsqueeze(dim=2).to(device) # our_time_embedding = [batch, frame, 1, channel]
        # # conv for time
        # features += time_embedding # feature = [batch, frame, dim, channel]

        features = features.permute(0, 1, 3, 2) # [8, 10, 64, 80][batch, L_frame, n_point, Channel]

        output = self.transformer(xyzs, features)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]

        output = self.mlp_head(output)
        output = torch.concat((output, torch.tensor(txt_feature).to(device).to(torch.float32)), dim=1)
        output = self.mlp_fusion(output)


        return output

class PSTTransformerFusionTxtMINE(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 dim, depth, heads, dim_head, dropout1,                                 # transformer
                 mlp_dim, num_classes, dropout2):                                       # output
        super().__init__()
        self.PositionalEncoding = PositionalEncoding(64,10)
        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, 80),
        )

        self.mlp_fusion1 = nn.Sequential(
            nn.GELU(),
            nn.Linear(82, 200),
            nn.Linear(200, 128)
        )

        self.mlp_fusion2 = nn.Sequential(
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, input, txt_feature):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L_frame, Channel, n_point]

        # # 合并降采样
        # pe_reshaped = pe.view(pe.shape[0], pe.shape[1] // 2, 2)
        # # 取均值
        # pe_new = pe_reshaped.mean(dim=2)
        # # 生成序列变化长度
        # sequence = torch.arange(1, pe_new.shape[1] + 1).unsqueeze(0).repeat(pe_new.shape[0], 1).to(device)
        # 生成时序差异的时间特征
        # pe_sequence = pe_new + sequence
        # 扩充维度
        # pe_sequence = pe_new.to(torch.float32).unsqueeze(dim=2).unsqueeze(dim=3)




        # time_embedding = self.PositionalEncoding(features).unsqueeze(dim=2).to(device) # our_time_embedding = [batch, frame, 1, channel]
        # # conv for time
        # features += time_embedding # feature = [batch, frame, dim, channel]

        features = features.permute(0, 1, 3, 2) # [8, 10, 64, 80][batch, L_frame, n_point, Channel]

        output, attn_score = self.transformer(xyzs, features)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]

        output = self.mlp_head(output)
        output = torch.concat((output, torch.tensor(txt_feature).to(device).to(torch.float32)), dim=1)
        output_embedded = self.mlp_fusion1(output)
        output = self.mlp_fusion2(output_embedded)


        return output, output_embedded, attn_score, xyzs

