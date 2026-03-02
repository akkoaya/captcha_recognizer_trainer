"""
识别头 (Head)

负责将骨干网络输出的特征序列转换为最终的字符预测。

提供两种识别头:
  1. CTCHead:       简单高效，用 CTC 损失函数，适合大多数验证码
  2. AttentionHead: 更强大，用 Transformer 注意力机制，适合复杂场景
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  CTC 识别头
# ═══════════════════════════════════════════════════════════════


class CTCHead(nn.Module):
    """
    CTC 识别头: 将特征序列映射到字符概率分布

    结构极其简单：就是一个全连接层 (Linear)。
    复杂度全在 CTC Loss 中——它负责自动对齐输入序列和输出字符。

    输入: [B, T, D]       T 个时间步的特征
    输出: [B, T, vocab]   每个时间步在所有字符上的概率分布
    """

    def __init__(self, feature_dim: int, vocab_size: int):
        """
        Args:
            feature_dim: 输入特征维度 (骨干网络输出维度)
            vocab_size:  词表大小 (包含 CTC blank 符号)
        """
        super().__init__()
        self.fc = nn.Linear(feature_dim, vocab_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T, D] 骨干网络输出的特征序列

        Returns:
            [B, T, vocab_size] 每个时间步的字符概率 (logits)
        """
        return self.fc(features)


# ═══════════════════════════════════════════════════════════════
#  Attention 识别头 (Transformer Decoder)
# ═══════════════════════════════════════════════════════════════


class PositionalEncoding(nn.Module):
    """
    位置编码: 为序列中的每个位置添加独特的位置信息

    Transformer 本身没有位置概念（不像 RNN 有先后顺序），
    所以需要额外注入位置信息。

    使用正弦/余弦函数生成固定的位置编码:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    不同位置的编码模式不同，模型可以通过这些模式感知位置。
    """

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维度用 cos
        pe = pe.unsqueeze(0)                             # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将位置编码加到输入上"""
        return x + self.pe[:, : x.size(1)]


class AttentionHead(nn.Module):
    """
    Attention 识别头: 基于 Transformer Decoder 的序列生成

    训练时: 用 Teacher Forcing (喂入真实标签作为解码器输入)
    推理时: 自回归生成 (逐个字符生成，每个字符依赖前面的输出)

    ┌──────────────────────────────────────────────────┐
    │  编码器输出 (骨干网络特征):                         │
    │  "这张图片有哪些视觉特征"                           │
    │       │                                          │
    │       ▼                                          │
    │  ┌──────────────────────┐                        │
    │  │ Transformer Decoder   │                       │
    │  │                      │                        │
    │  │  Cross-Attention:    │ ← "哪些视觉特征对应      │
    │  │  关注编码器输出       │    当前要预测的字符?"     │
    │  │                      │                        │
    │  │  Self-Attention:     │ ← "已经预测了什么字符?    │
    │  │  关注已生成的字符     │    下一个应该是什么?"     │
    │  └──────────┬───────────┘                        │
    │             │                                    │
    │             ▼                                    │
    │       输出字符概率                                 │
    └──────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        feature_dim: int,
        vocab_size: int,
        max_len: int = 50,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dim: 特征维度
            vocab_size:  词表大小
            max_len:     最大生成长度
            num_heads:   注意力头数 (多头注意力)
            num_layers:  Transformer 解码器层数
            dropout:     Dropout 比率
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.sos_index = 1    # 与 Tokenizer 保持一致
        self.eos_index = 2

        # 字符嵌入层: 将字符索引转为连续向量
        self.embedding = nn.Embedding(vocab_size, feature_dim, padding_idx=0)

        # 位置编码: 告诉模型每个字符在序列中的位置
        self.pos_encoding = PositionalEncoding(feature_dim, max_len)

        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出投影: 从特征空间映射到词表空间
        self.output_proj = nn.Linear(feature_dim, vocab_size)

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        生成因果掩码 (Causal Mask):
        防止解码器 "偷看" 未来的字符。

        例如预测第 3 个字符时，只能看到第 1、2 个字符，
        不能看到第 4、5 个字符。

        掩码矩阵 (True = 被屏蔽):
          0 1 1 1
          0 0 1 1
          0 0 0 1
          0 0 0 0
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.bool()

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, D] 编码器输出 (骨干网络特征)
            targets:  [B, S] 目标序列 (训练时使用，包含 SOS/EOS)

        Returns:
            [B, S-1, vocab_size] 字符概率分布
        """
        # 训练模式: Teacher Forcing
        # 输入: [SOS, a, 3, K, p]  目标: [a, 3, K, p, EOS]
        tgt_input = targets[:, :-1]      # 去掉最后一个 token
        tgt_emb = self.embedding(tgt_input)
        tgt_emb = self.pos_encoding(tgt_emb)

        # 因果掩码: 不能看到后面的字符
        causal_mask = self._generate_causal_mask(tgt_input.size(1), tgt_input.device)

        # Transformer 解码
        output = self.decoder(
            tgt=tgt_emb,
            memory=features,
            tgt_mask=causal_mask,
        )

        return self.output_proj(output)

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        推理模式: 自回归生成

        从 SOS 开始，逐个预测字符，直到输出 EOS 或达到最大长度。

        Args:
            features: [B, T, D] 编码器输出

        Returns:
            [B, max_len] 预测的字符索引序列
        """
        batch_size = features.size(0)
        device = features.device

        # 初始输入: 全是 SOS
        input_ids = torch.full(
            (batch_size, 1), self.sos_index, dtype=torch.long, device=device
        )

        for _ in range(self.max_len):
            tgt_emb = self.embedding(input_ids)
            tgt_emb = self.pos_encoding(tgt_emb)
            causal_mask = self._generate_causal_mask(input_ids.size(1), device)

            output = self.decoder(
                tgt=tgt_emb,
                memory=features,
                tgt_mask=causal_mask,
            )

            # 取最后一个时间步的预测
            logits = self.output_proj(output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 所有序列都已生成 EOS，提前结束
            if (next_token == self.eos_index).all():
                break

        return input_ids[:, 1:]   # 去掉开头的 SOS
