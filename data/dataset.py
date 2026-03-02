"""
数据集模块

将图片文件夹中的验证码图片加载为 PyTorch Dataset。
图片文件名格式: 标签_任意内容.后缀，如 a3Kp_001.png

主要流程:
  1. 扫描文件夹中的所有图片
  2. 从文件名中提取标签
  3. 加载图片 → 缩放 → 数据增强 → 归一化 → 张量
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from data.tokenizer import Tokenizer


class CaptchaDataset(Dataset):
    """验证码数据集：从图片文件夹加载数据"""

    # 支持的图片格式
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(
        self,
        data_dir: str,
        tokenizer: Tokenizer,
        img_height: int = 64,
        img_width: int = 160,
        img_channel: int = 1,
        transform=None,
        label_sep: str = "_",
    ):
        """
        Args:
            data_dir:    图片文件夹路径
            tokenizer:   字符编解码器
            img_height:  目标图片高度
            img_width:   目标图片宽度
            img_channel: 图片通道数 (1=灰度, 3=彩色)
            transform:   Albumentations 数据增强
            label_sep:   文件名中标签与其余部分的分隔符
        """
        self.tokenizer = tokenizer
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel
        self.transform = transform
        self.label_sep = label_sep

        # 扫描所有图片文件
        self.samples = []
        for filename in sorted(os.listdir(data_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.IMAGE_EXTENSIONS:
                filepath = os.path.join(data_dir, filename)
                # 从文件名中提取标签: "a3Kp_001.png" → "a3Kp"
                label = os.path.splitext(filename)[0].split(label_sep)[0]
                self.samples.append((filepath, label))

        if not self.samples:
            raise FileNotFoundError(f"在 {data_dir} 中未找到图片文件")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filepath, label_text = self.samples[idx]

        # ── 1. 加载图片 ──
        if self.img_channel == 1:
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            raise IOError(f"无法读取图片: {filepath}")

        # ── 2. 缩放到统一尺寸 ──
        image = cv2.resize(image, (self.img_width, self.img_height))

        # ── 3. 数据增强 (仅训练时) ──
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # ── 4. 转为浮点数并归一化到 [0, 1] ──
        image = image.astype(np.float32) / 255.0

        # ── 5. 转为 PyTorch 张量 ──
        #   灰度图: [H, W] → [1, H, W]
        #   彩色图: [H, W, 3] → [3, H, W]
        if self.img_channel == 1:
            image = torch.FloatTensor(image).unsqueeze(0)   # 添加通道维度
        else:
            image = torch.FloatTensor(image).permute(2, 0, 1)  # HWC → CHW

        # ── 6. 编码标签 ──
        label_text_lower = label_text.lower()
        encoded_label = self.tokenizer.encode(label_text_lower)

        return image, encoded_label, label_text_lower


def ctc_collate_fn(batch):
    """
    CTC 模式的批次整理函数

    CTC Loss 需要:
      - images:         [B, C, H, W]  图片张量
      - targets:        [所有标签拼接]  一维张量
      - target_lengths: [B]           每个标签的长度

    为什么 targets 是拼接的?
      PyTorch 的 CTCLoss 支持两种 target 格式:
      1. 拼接的一维张量 + 长度 (更高效)
      2. 填充的二维张量 + 长度
      这里用第一种。
    """
    images, labels, texts = zip(*batch)

    # 图片堆叠为 [B, C, H, W]
    images = torch.stack(images)

    # 标签拼接为一维张量
    target_lengths = torch.LongTensor([len(label) for label in labels])
    targets = torch.LongTensor([idx for label in labels for idx in label])

    return images, targets, target_lengths, list(texts)


def attention_collate_fn(batch):
    """
    Attention 模式的批次整理函数

    Attention 需要完整的目标序列 (带 SOS/EOS):
      输入:   [SOS, a, 3, K, p]        → 解码器输入
      目标:   [a, 3, K, p, EOS]        → 计算交叉熵损失
      填充到相同长度，用 PAD(0) 补齐。
    """
    images, labels, texts = zip(*batch)

    images = torch.stack(images)

    # 找到最长标签 + 2 (SOS + EOS)
    max_len = max(len(label) for label in labels) + 2

    # 构建目标序列: [SOS, char1, char2, ..., EOS, PAD, PAD, ...]
    targets = torch.zeros(len(labels), max_len, dtype=torch.long)  # 0 = PAD
    target_lengths = []
    for i, label in enumerate(labels):
        targets[i, 0] = 1                     # SOS index = 1
        for j, idx in enumerate(label):
            targets[i, j + 1] = idx
        targets[i, len(label) + 1] = 2        # EOS index = 2
        target_lengths.append(len(label) + 2)  # 包含 SOS 和 EOS

    target_lengths = torch.LongTensor(target_lengths)

    return images, targets, target_lengths, list(texts)
