"""
完整识别模型

组合骨干网络 (Backbone) + 识别头 (Head) 为完整的端到端模型。

两种模式:
  - CTC 模式:       Backbone → CTCHead → CTC Loss
  - Attention 模式:  Backbone → AttentionHead → CrossEntropy Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import ResNetBackbone
from model.head import CTCHead, AttentionHead


class CaptchaRecognizer(nn.Module):
    """
    验证码识别模型: 从图片中识别文字

    支持两种识别模式:
      1. CTC 模式 (推荐初学者使用):
         - 优点: 结构简单，不需要对齐标签，训练快
         - 缺点: 假设各时间步独立，无法建模字符间依赖
         - 适用: 大多数验证码场景

      2. Attention 模式:
         - 优点: 能建模字符间依赖，理论上精度更高
         - 缺点: 训练更慢，需要更多数据
         - 适用: 复杂验证码、自然场景文字

    使用示例:
        model = CaptchaRecognizer(config, vocab_size=39)

        # 训练
        loss = model(images, targets, target_lengths)
        loss.backward()

        # 推理
        predictions = model.predict(images)
    """

    def __init__(self, config: dict, vocab_size: int):
        """
        Args:
            config:     配置字典 (从 config.yaml 加载)
            vocab_size: 词表大小 (字符数 + 特殊标记数)
        """
        super().__init__()

        model_config = config["model"]
        data_config = config["data"]

        self.head_type = model_config["head"]           # "ctc" 或 "attention"
        feature_dim = model_config.get("feature_dim", 256)
        in_channels = data_config.get("image_channel", 1)
        max_label_length = data_config.get("max_label_length", 10)

        # ── 骨干网络: 提取图像特征 ──
        self.backbone = ResNetBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )

        # ── 识别头: 特征 → 字符 ──
        if self.head_type == "ctc":
            self.head = CTCHead(feature_dim, vocab_size)
        elif self.head_type == "attention":
            self.head = AttentionHead(
                feature_dim=feature_dim,
                vocab_size=vocab_size,
                max_len=max_label_length + 2,   # +2 for SOS/EOS
            )
        else:
            raise ValueError(f"不支持的识别头类型: {self.head_type}")

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        权重初始化: 使用 Kaiming 初始化 (适配 GELU/ReLU 激活函数)

        良好的初始化让每层输出的方差保持一致，
        避免训练初期梯度爆炸或消失。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        训练时的前向传播: 计算损失

        Args:
            images:         [B, C, H, W] 输入图片
            targets:        标签 (格式因模式而异，见下方说明)
            target_lengths: [B] 每个标签的有效长度

        Returns:
            标量损失值
        """
        # 骨干网络提取特征
        features = self.backbone(images)    # [B, T, D]

        if self.head_type == "ctc":
            return self._ctc_forward(features, targets, target_lengths)
        else:
            return self._attention_forward(features, targets)

    def _ctc_forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        CTC 模式的前向传播

        CTC Loss 的输入:
          - log_probs:      [T, B, vocab]  每个时间步的对数概率 (注意是 T 在前)
          - targets:         [所有标签拼接]   字符索引
          - input_lengths:   [B]            每个样本的时间步数 (都相同)
          - target_lengths:  [B]            每个标签的字符数
        """
        logits = self.head(features)                         # [B, T, vocab]
        log_probs = F.log_softmax(logits, dim=2)             # 转为对数概率
        log_probs = log_probs.permute(1, 0, 2)               # [T, B, vocab]

        batch_size = features.size(0)
        input_lengths = torch.full(
            (batch_size,), log_probs.size(0), dtype=torch.long, device=features.device
        )

        loss = F.ctc_loss(
            log_probs, targets, input_lengths, target_lengths,
            blank=0, zero_infinity=True,
        )
        return loss

    def _attention_forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention 模式的前向传播

        使用 Teacher Forcing:
          解码器输入: [SOS, a, 3, K, p]    (真实标签前面加 SOS)
          解码器目标: [a, 3, K, p, EOS]    (真实标签后面加 EOS)
          用交叉熵损失对比预测和目标
        """
        logits = self.head(features, targets)               # [B, S-1, vocab]

        # 目标: targets 去掉第一个 SOS
        tgt_output = targets[:, 1:]                          # [B, S-1]

        # 交叉熵损失 (忽略 PAD 位置)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=0,     # PAD index = 0
        )
        return loss

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> list:
        """
        推理: 从图片预测字符序列

        Args:
            images: [B, C, H, W] 输入图片

        Returns:
            list[list[int]]: 每张图片的预测索引序列
        """
        self.eval()
        features = self.backbone(images)

        if self.head_type == "ctc":
            logits = self.head(features)                    # [B, T, vocab]
            preds = logits.argmax(dim=2)                    # [B, T] 贪心解码
            return preds.cpu().tolist()
        else:
            preds = self.head.predict(features)             # [B, max_len]
            return preds.cpu().tolist()
