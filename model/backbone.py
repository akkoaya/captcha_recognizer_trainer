"""
骨干网络 (Backbone)

负责从输入图片中提取特征。
将 [B, C, H, W] 的图片转换为 [B, T, D] 的特征序列。
  B = 批大小
  T = 序列长度 (由图片宽度决定)
  D = 特征维度

使用残差卷积网络 (ResNet 风格)，核心思想:
  1. 卷积层提取局部特征 (边缘、纹理、笔画)
  2. 池化层逐步缩小高度，保留宽度方向的分辨率
  3. 残差连接让梯度畅通无阻，网络更容易训练
  4. 最终将高度压缩为 1，宽度方向作为时间步
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    基础卷积块: Conv → BatchNorm → GELU

    这是构成所有网络的最小单元。每次卷积操作后:
    - BatchNorm: 标准化输出分布，加速训练
    - GELU: 激活函数，引入非线性（比 ReLU 更平滑）
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,    # same padding，保持尺寸不变
            bias=False,                   # 有 BN 时不需要 bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    残差块: 让网络可以很深而不会梯度消失

    核心公式: output = activation(F(x) + shortcut(x))

    其中:
    - F(x) 是两层卷积学到的 "残差"
    - shortcut(x) 是恒等映射 (或 1x1 卷积调整维度)
    - 两者相加后再激活

    为什么有效?
    如果某一层不需要学习新东西，F(x) 只需要学到 0，
    输出就等于输入 x。这比从头学 "输出=输入" 简单得多。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # 主路径: 两层卷积
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        # 捷径 (Shortcut):
        #   通道数不变 → 直接加 (恒等映射)
        #   通道数改变 → 用 1x1 卷积调整维度
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主路径
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 残差连接: 主路径 + 捷径
        out = out + self.shortcut(x)

        return self.act(out)


class ResNetBackbone(nn.Module):
    """
    骨干网络: 基于 ResNet 的特征提取器

    网络结构:
    ┌────────────────────────────────────────────────────────────┐
    │  输入: [B, 1, 64, 160]                                     │
    │                                                            │
    │  Stem (入口层):                                             │
    │    Conv(1→32) + Conv(32→32) + MaxPool(2×2)                 │
    │    → [B, 32, 32, 80]                                       │
    │                                                            │
    │  Stage1: ResBlock(32→64) + MaxPool(2×2)                    │
    │    → [B, 64, 16, 40]                                       │
    │                                                            │
    │  Stage2: ResBlock(64→128) + MaxPool(高度减半，宽度不变)       │
    │    → [B, 128, 8, 40]                                       │
    │                                                            │
    │  Stage3: ResBlock(128→256) + MaxPool(高度减半，宽度不变)      │
    │    → [B, 256, 4, 40]                                       │
    │                                                            │
    │  压缩高度: 对高度维度取平均                                    │
    │    → [B, 256, 40]                                          │
    │                                                            │
    │  转置为序列: [B, 40, 256]                                    │
    │    T=40 个时间步, 每步 D=256 维特征                           │
    └────────────────────────────────────────────────────────────┘

    为什么后两个 Stage 只缩小高度不缩小宽度?
      因为宽度方向对应字符的水平位置，保留宽度分辨率才能区分不同位置的字符。
      高度方向的信息最终被压缩掉，因为同一列的所有像素属于同一个字符。
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 256):
        """
        Args:
            in_channels: 输入通道数 (1=灰度, 3=彩色)
            feature_dim: 输出特征维度
        """
        super().__init__()

        self.features = nn.Sequential(
            # ── Stem: 初始特征提取 ──
            ConvBlock(in_channels, 32, kernel_size=3),
            ConvBlock(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),        # H/2, W/2

            # ── Stage 1 ──
            ResidualBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),        # H/4, W/4

            # ── Stage 2 ──
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/8, W 不变

            # ── Stage 3 ──
            ResidualBlock(128, feature_dim),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/16, W 不变
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图片 [B, C, H, W]

        Returns:
            特征序列 [B, T, D]
            T = W / 4 (宽度被缩小了4倍)
            D = feature_dim
        """
        x = self.features(x)       # [B, feature_dim, H', W/4]
        x = x.mean(dim=2)          # [B, feature_dim, W/4]  对高度取平均
        x = x.permute(0, 2, 1)     # [B, W/4, feature_dim] = [B, T, D]
        return x
