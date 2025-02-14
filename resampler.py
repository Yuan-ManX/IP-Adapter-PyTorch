import math
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


# ===========================
# 前馈神经网络（Feed-Forward Network, FFN）
# ===========================

def FeedForward(dim, mult=4):
    """
    构建一个前馈神经网络（FFN）层，通常用于Transformer模型中。

    该网络由以下部分组成：
    1. 层归一化（LayerNorm）
    2. 线性变换，将维度从 `dim` 扩展到 `inner_dim`（通常是输入维度的4倍）
    3. 高斯误差线性单元（GELU）激活函数
    4. 线性变换，将维度从 `inner_dim` 恢复到 `dim`

    Args:
        dim (int): 输入和输出的维度大小。
        mult (int, optional): 内部维度相对于输入维度的倍数，默认为4。

    Returns:
        nn.Sequential: 包含上述层的前馈神经网络。
    """
    # 计算内部维度
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim), # 对输入进行层归一化
        nn.Linear(dim, inner_dim, bias=False), # 线性变换，扩展维度
        nn.GELU(), # 应用GELU激活函数
        nn.Linear(inner_dim, dim, bias=False), # 线性变换，恢复原始维度
    )


# ===========================
# 张量重塑函数
# ===========================

def reshape_tensor(x, heads):
    """
    重塑输入张量以适应多头注意力机制。

    该函数将输入张量从 (batch_size, length, width) 的形状重塑为 (batch_size * heads, length, dim_per_head)，
    以便进行多头注意力的计算。

    Args:
        x (torch.Tensor): 输入张量，形状为 (batch_size, length, width)。
        heads (int): 多头注意力的头数。

    Returns:
        torch.Tensor: 重塑后的张量，形状为 (batch_size * heads, length, dim_per_head)。
    """
    # 分离输入张量的维度：批次大小、长度和宽度
    bs, length, width = x.shape
    # 将张量重塑为 (batch_size, length, heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # 转置维度为 (batch_size, heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # 重新调整形状为 (batch_size * heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


# ===========================
# PerceiverAttention 类
# ===========================

class PerceiverAttention(nn.Module):
    """
    Perceiver模型的注意力机制实现。

    该类实现了Perceiver模型中的注意力机制，允许模型处理高维输入（如图像），
    并通过交叉注意力机制与潜在表示（latents）进行交互。

    Args:
        dim (int): 输入和潜在表示的维度大小。
        dim_head (int, optional): 每个注意力头的维度大小，默认为64。
        heads (int, optional): 多头注意力的头数，默认为8。
    """
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        # 缩放因子，用于缩放注意力得分
        self.scale = dim_head**-0.5
        # 每个注意力头的维度大小
        self.dim_head = dim_head
        # 多头注意力的头数
        self.heads = heads
        # 计算内部维度
        inner_dim = dim_head * heads

        # 定义层归一化层
        # 对输入进行层归一化
        self.norm1 = nn.LayerNorm(dim)
        # 对潜在表示进行层归一化
        self.norm2 = nn.LayerNorm(dim)

        # 定义线性变换层，用于计算查询（Q）、键（K）和值（V）
        self.to_q = nn.Linear(dim, inner_dim, bias=False) # 查询的线性变换
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False) # 键和值的线性变换
        self.to_out = nn.Linear(inner_dim, dim, bias=False) # 输出线性变换

    def forward(self, x, latents):
        """
        前向传播方法，执行Perceiver的注意力机制计算。

        Args:
            x (torch.Tensor): 输入的图像特征，形状为 (batch_size, n1, D)。
            latents (torch.Tensor): 潜在表示特征，形状为 (batch_size, n2, D)。

        Returns:
            torch.Tensor: 经过注意力机制处理后的输出张量，形状为 (batch_size, n2, D)。
        """
        # 对输入张量进行层归一化
        x = self.norm1(x)
        # 对潜在表示张量进行层归一化
        latents = self.norm2(latents)

        # 获取潜在表示的批次大小和长度
        b, l, _ = latents.shape

        # 计算查询（Q）向量
        q = self.to_q(latents)
        # 将输入和潜在表示连接起来，作为键（K）和值（V）的输入
        kv_input = torch.cat((x, latents), dim=-2)
        # 计算键（K）和值（V）向量
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # 重塑查询、键和值张量以适应多头注意力机制
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # 计算注意力得分
        # 使用缩放因子进行缩放，以确保梯度稳定
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        # 计算注意力权重
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        # 应用softmax函数，将注意力权重转换为概率分布
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        # 通过注意力权重对值进行加权求和，得到输出
        out = weight @ v

        # 重塑输出张量的形状为 (batch_size, n_heads, length, dim_per_head)
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        # 通过线性变换层进行输出投影
        return self.to_out(out)


class Resampler(nn.Module):
    """
    重采样器（Resampler）类，用于在特征空间中执行重采样操作。

    该类通过使用潜在表示（latents）和多头注意力机制，从输入特征中提取和重采样信息。
    它可以用于例如图像特征的上采样、下采样或跨尺度特征融合等任务。

    Args:
        dim (int, optional): 潜在表示和注意力层的维度大小，默认为1024。
        depth (int, optional): 注意力层和前馈层的堆叠深度，默认为8。
        dim_head (int, optional): 每个注意力头的维度大小，默认为64。
        heads (int, optional): 多头注意力的头数，默认为16。
        num_queries (int, optional): 查询（queries）的数量，默认为8。
        embedding_dim (int, optional): 输入嵌入特征的维度大小，默认为768。
        output_dim (int, optional): 输出特征的维度大小，默认为1024。
        ff_mult (int, optional): 前馈层内部维度相对于输入维度的倍数，默认为4。
        max_seq_len (int, optional): 最大序列长度，默认为257（CLIP tokens + CLS token）。
        apply_pos_emb (bool, optional): 是否应用位置嵌入，默认为False。
        num_latents_mean_pooled (int, optional): 从序列的均值池化表示中派生的潜在表示数量，默认为0。
    """
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        # 如果需要应用位置嵌入，则初始化位置嵌入层
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        # 初始化潜在表示（latents），形状为 (1, num_queries, dim)，并使用正态分布进行初始化
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        # 定义一个线性变换层，用于将嵌入特征的维度从 embedding_dim 转换为 dim
        self.proj_in = nn.Linear(embedding_dim, dim)

        # 定义一个线性变换层，用于将潜在表示的维度从 dim 转换为 output_dim
        self.proj_out = nn.Linear(dim, output_dim)
        # 定义一个层归一化层，用于对输出进行归一化
        self.norm_out = nn.LayerNorm(output_dim)

        # 如果需要从均值池化的序列中派生出潜在表示，则初始化相应的层
        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),  # 对输入进行层归一化
                nn.Linear(dim, dim * num_latents_mean_pooled),  # 线性变换，扩展维度
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),  # 重排张量形状
            )
            if num_latents_mean_pooled > 0
            else None
        )

        # 初始化多层感知机列表，用于堆叠多个处理层
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # 对于每一层，添加一个包含注意力层和前馈层的模块列表
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads), # 多头注意力层
                        FeedForward(dim=dim, mult=ff_mult), # 前馈神经网络层
                    ]
                )
            )

    def forward(self, x):
        """
        前向传播方法，执行重采样操作。

        Args:
            x (torch.Tensor): 输入嵌入特征，形状为 (batch_size, sequence_length, embedding_dim)。

        Returns:
            torch.Tensor: 经过重采样后的输出特征，形状为 (batch_size, num_queries, output_dim)。
        """
        # 如果需要应用位置嵌入，则生成位置嵌入并将其添加到输入特征中
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        # 重复潜在表示，以匹配输入的批次大小，形状变为 (batch_size, num_queries, dim)
        latents = self.latents.repeat(x.size(0), 1, 1)

        # 将输入嵌入特征的维度从 embedding_dim 转换为 dim
        x = self.proj_in(x)

        # 如果需要从均值池化的序列中派生出潜在表示，则进行相应的处理
        if self.to_latents_from_mean_pooled_seq:
            # 对输入特征进行均值池化，得到形状为 (batch_size, embedding_dim)
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            # 从均值池化的序列中派生出潜在表示，形状为 (batch_size, num_latents_mean_pooled, dim)
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            # 将派生的潜在表示与原始潜在表示连接起来，形状为 (batch_size, num_queries + num_latents_mean_pooled, dim)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        # 遍历每一层，包括注意力层和前馈层
        for attn, ff in self.layers:
            # 执行注意力机制，将输入特征与潜在表示结合，更新潜在表示
            latents = attn(x, latents) + latents
            # 通过前馈神经网络进一步处理潜在表示
            latents = ff(latents) + latents

        # 将潜在表示的维度从 dim 转换为 output_dim
        latents = self.proj_out(latents)
        # 对输出进行层归一化
        return self.norm_out(latents)


def masked_mean(t, *, dim, mask=None):
    """
    对输入张量进行带掩码的均值计算。

    Args:
        t (torch.Tensor): 输入张量。
        dim (int): 要进行均值计算的维度。
        mask (torch.Tensor, optional): 掩码张量，用于指示哪些元素参与计算。默认为None。

    Returns:
        torch.Tensor: 带掩码的均值结果。
    """
    if mask is None:
        # 如果没有提供掩码，则直接返回输入张量的均值
        return t.mean(dim=dim)
    
    # 计算掩码中1的数量，作为分母
    denom = mask.sum(dim=dim, keepdim=True)
    # 重塑掩码张量，使其与输入张量形状兼容
    mask = rearrange(mask, "b n -> b n 1")
    # 使用掩码对输入张量进行掩码填充，掩码为0的位置填充为0
    masked_t = t.masked_fill(~mask, 0.0)

    # 计算带掩码的均值
    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)
