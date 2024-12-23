import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper functions
def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d
def divisible_by(numer, denom):
    return (numer % denom) == 0
# tensor helpers
def grid_sample_1d(feats, grid, *args, **kwargs):
    # does 1d grid sample by reshaping it to 2d
    grid = rearrange(grid, '... -> ... 1 1')
    grid = F.pad(grid, (1, 0), value=0.)
    feats = rearrange(feats, '... -> ... 1')
    out = F.grid_sample(feats, grid, **kwargs)
    return rearrange(out, '... 1 -> ...')
def normalize_grid(arange, dim=1, out_dim=-1):
    # normalizes 1d sequence to range of -1 to 1
    n = arange.shape[-1]
    return 2.0 * arange / max(n - 1, 1) - 1.0
import einops

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x) + x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.spatial_op = nn.Linear(3, dim_head, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, xyzs, features):
        b, l, n, _, h = *features.shape, self.heads # attention模块接受xyzs空间坐标 [B, L, n, 3]和features [batch, frame, dim, channel]

        # for features
        norm_features = self.norm(features) # 对输入特征进行层归一化
        qkv = self.to_qkv(norm_features).chunk(3, dim = -1) # 将归一化后的特征通过to_qkv映射为qkv矩阵
        q, k, v = map(lambda t: rearrange(t, 'b l n (h d) -> b h (l n) d', h = h), qkv)                            # [b, h, m, d]变换形状，用于multi-head

        # W_l = pe_sequence.unsqueeze(1).expand(b, h, l, 1, 1)
        # for features ## qkv caculate
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale                                      # [b, h, m, m]计算查询和键之间的点积
        attn = dots.softmax(dim=-1)                                                                                # 对点积结果进行 softmax，得到注意力权重

        v = einsum('b h i j, b h j d -> b h i d', attn, v)                                                   # [b, h, m, d]计算加权值

        # # 加权代码
        # v = rearrange(v, 'b h (l n) d -> b h l n d', l=l)  # Reshape to expose l dimension
        # v = v * W_l  # Apply weight
        # v = rearrange(v, 'b h l n d -> b h (l n) d')  # Restore original shape

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

class Attention_getattn(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.spatial_op = nn.Linear(3, dim_head, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, xyzs, features):
        b, l, n, _, h = *features.shape, self.heads # attention模块接受xyzs空间坐标 [B, L, n, 3]和features [batch, frame, dim, channel]

        # for features
        norm_features = self.norm(features) # 对输入特征进行层归一化
        qkv = self.to_qkv(norm_features).chunk(3, dim = -1) # 将归一化后的特征通过to_qkv映射为qkv矩阵
        q, k, v = map(lambda t: rearrange(t, 'b l n (h d) -> b h (l n) d', h = h), qkv)                            # [b, h, m, d]变换形状，用于multi-head

        # W_l = pe_sequence.unsqueeze(1).expand(b, h, l, 1, 1)
        # for features ## qkv caculate
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale                                      # [b, h, m, m]计算查询和键之间的点积
        attn = dots.softmax(dim=-1)                                                                                # 对点积结果进行 softmax，得到注意力权重

        v = einsum('b h i j, b h j d -> b h i d', attn, v)                                                   # [b, h, m, d]计算加权值

        # # 加权代码
        # v = rearrange(v, 'b h (l n) d -> b h l n d', l=l)  # Reshape to expose l dimension
        # v = v * W_l  # Apply weight
        # v = rearrange(v, 'b h l n d -> b h (l n) d')  # Restore original shape

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
        return out + features, attn

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.attention_list = []
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                Attention_getattn(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, xyzs, features):
        for attn, ff in self.layers:
            features, attn_score = attn(xyzs, features) # get attn
            features = ff(features)
            self.attention_list.append(attn_score)
        return features, self.attention_list

