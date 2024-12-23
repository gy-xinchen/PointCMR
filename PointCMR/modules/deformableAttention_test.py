# -*- coding: utf-8 -*-
# @Time    : 2024/11/18 13:08
# @Author  : yuan
# @File    : deformableAttention_test.py
import torch
import torch.nn.functional as F
from torch import nn, einsum
import einops
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads # 矩阵乘法 inner = dimhead * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) # 这个代码用于同时计算qkv
        self.spatial_op = nn.Linear(3, dim_head, bias = False) # 用于映射点云空间维度

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() # 输出层的线性操作

    def forward(self, xyzs, features):
        b, l, n, _, h = *features.shape, self.heads         # [batch, L_frame, n_point, Channel, heads]

        # for features
        norm_features = self.norm(features) # 对输入特征进行层归一化
        qkv = self.to_qkv(norm_features).chunk(3, dim = -1) # 将归一化后的特征通过to_qkv映射为qkv矩阵
        q, k, v = map(lambda t: rearrange(t, 'b l n (h d) -> b h (l n) d', h = h), qkv)    # 拆分head [b, l, n, C(h*d)] -->[b, h, l*n, d]
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale                      # 计算查询和键之间的点积 [b, h, l*n, d]*[b, h, l*n, d]--> [b, h, m, m]
        attn = dots.softmax(dim=-1)                                                                # 对点积结果进行 softmax，得到注意力权重
        v = einsum('b h i j, b h j d -> b h i d', attn, v)                                   # 计算加权后值 [b, h, m, m]*[b, h, l*n, d]-->[b, h, l*n, d]并且恢复head-->[b,l,n,C]

        # for xyzs
        xyzs_flatten = rearrange(xyzs, 'b l n d -> b (l n) d')                                                     # [b, m, 3]压缩xyzs空间坐标以用于注意力计算
        delta_xyzs = torch.unsqueeze(input=xyzs_flatten, dim=1) - torch.unsqueeze(input=xyzs_flatten, dim=2)       # [b, m, m, 3]计算空间坐标之间的差异，包含了每一对空间位置之间的坐标差异
        attn = torch.unsqueeze(input=attn, dim=4)                                                                   # [b, h, m, m, 1]表示每个位置对其他位置的注意力分数
        delta_xyzs = torch.unsqueeze(input=delta_xyzs, dim=1)                                                       # [b, 1, m, m, 3]表示每个位置对其他位置的空间位移
        delta_xyzs = torch.sum(input=attn*delta_xyzs, dim=3, keepdim=False)                                         # [b, h, m, 3]表示每个位置对其他位置的位移在不同注意力头下的加权值
        displacement_features = self.spatial_op(delta_xyzs)                                                         # [b, h, m, d]为加权位移特征应用线性变换

        out = v + displacement_features                                                                             # 将注意力特征v与加权位移特征displacement_features结合

        out = rearrange(out, 'b h m d -> b m (h d)')
        out =  self.to_out(out)
        out = rearrange(out, 'b (l n) d -> b l n d', l=l, n=n)
        return out + features
class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class DAttentionBaseline(nn.Module):

    def __init__(
            self, q_size, n_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor, use_pe, dwc_pe,
            no_off, fixed_pe, ksize, log_cpb
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0
        # q_size 和 kv_size分别代表查询 (query) 特征图和键值 (key-value) 特征图的空间尺寸
        # n_heads 和 n_head_channels 多头注意力中的头数 每个头的通道数， 总通道数 nc = n_heads * n_head_channels
        # n_groups: 通道分组数
        # 每组通道数 n_group_channels = nc // n_groups
        # 每组中包含的头数 n_group_heads = n_heads // n_groups
        # stride 控制键值特征图 (kv) 与查询特征图 (q) 的下采样比例
        # attn_drop 和 proj_drop: 控制注意力得分和输出的 dropout 比例
        # offset_range_factor: 控制偏移的范围
        # use_pe 和 dwc_pe: 是否使用相对位置编码 (RPE) 以及是否采用深度可分离卷积实现 RPE
        # no_off: 是否禁用偏移 (offset)
        # fixed_pe 和 log_cpb: 决定位置编码的形式
        # ksize: 偏移卷积的核大小
        # log_cpb: 是否采用 log_cpb（对数相对位置偏移编码）

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        # 用于生成偏移量 (offset)，控制注意力区域的位置

        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        # 对输入特征图进行线性变换，分别得到查询 (query)、键 (key)、值 (value)。
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        # proj_out: 输出的特征图
        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        # 根据参数选择不同形式
        # dwc_pe=True: 使用深度可分离卷积实现位置编码
        # fixed_pe=True: 使用固定的可学习位置编码表
        # log_cpb=True: 借鉴 Swin-V2，采用对数编码
        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    # 获取参考点的函数，用于生成规范化的二维网格参考坐标，通常在注意力机制中用于计算偏移或定义采样点
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        # 创建二维网格参考点
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        # 堆叠坐标
        ref = torch.stack((ref_y, ref_x), -1)
        # 规范化坐标
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    # 生成一个用于查询位置的规范化网格坐标张量
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        # 查询点特征生成
        q = self.proj_q(x) # [B,C,H,W]
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels) # [14*5, 16,H, W]
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        # 偏移范围限制
        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')

        # 参考点生成
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        # 采样特征图生成
        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        # 键值生成
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        # 注意力计算
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        # 处理相对位置编码，可能在注意力分数中键入相对位置编码
        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    4.0)  # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    0.5)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
                                           g=self.n_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True)  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe

        # 输出处理
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)


class MyDeformableAttention(nn.Module):
    def __init__(self,
                 q_size, k_size, n_heads, n_head_channels, n_groups, stride, offset_range_factor, ksize):
        super().__init__()
        self.q_dim = q_size
        self.k_channel = k_size
        self.n_heads = n_heads
        self.scale = self.n_head_channels ** -0.5
        self.n_head_channels = n_head_channels
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.offset_range_factor = offset_range_factor
        self.stride = stride
        self.ksize = ksize
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        # 通过1x1卷积核实现qkv
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        # proj_out: 输出的特征图
        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

    @torch.no_grad()
    # 获取参考点的函数，用于生成规范化的二维网格参考坐标，通常在注意力机制中用于计算偏移或定义采样点
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        # 创建二维网格参考点
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        # 堆叠坐标
        ref = torch.stack((ref_y, ref_x), -1)
        # 规范化坐标
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    # 生成一个用于查询位置的规范化网格坐标张量
    def _get_q_grid(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, xyzs, features):
        b, f, d, c, h = *features.shape, self.n_heads

        # 查询点生成
        q = self.proj_q(features)
        q_off = einops.rearrange(q, 'b (g f) d c h -> (b g) f d c h', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

# 调整输入形状，适配2D注意力机制
batch_size = 14
L_frame = 10
n_point = 64
Channel = 80

# 创建随机输入张量
xyzs = torch.randn(batch_size, L_frame, n_point, 3) # [batch, L_frame, n_point, 3]
features = torch.randn(batch_size, L_frame, Channel, n_point)
features = features.permute(0, 2, 1, 3) # [batch, L_frame, n_point, Channel]

attention = DAttentionBaseline(q_size=(3,3), n_heads = 8, n_head_channels=10, n_groups=5,
                               attn_drop=0, proj_drop=0, stride=1, offset_range_factor=2, use_pe=False, dwc_pe=False,
                               no_off=False, fixed_pe=False, ksize=3, log_cpb=False)

# 前向传播
# output = attention(xyzs, features)
output = attention(features)

print(f"输出形状: {output.shape}")
