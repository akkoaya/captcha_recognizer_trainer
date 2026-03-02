# Captcha Recognizer

基于 **PyTorch** 的现代验证码识别框架。采用 ResNet + CTC/Attention 架构，支持混合精度训练、ONNX 导出、丰富的数据增强。代码注释详尽，适合深度学习初学者阅读和学习。

## 特性

- **现代架构**: ResNet 残差骨干网络 + CTC / Transformer Attention 双模式识别头
- **训练优化**: AdamW 优化器、OneCycleLR 学习率调度、混合精度 (AMP)、早停、梯度裁剪
- **数据增强**: 基于 [Albumentations](https://github.com/albumentations-team/albumentations)，高效且可组合
- **一键部署**: 导出 ONNX 格式，支持 ONNX Runtime / TensorRT 推理
- **开箱即用**: 内置验证码生成工具，无需额外数据即可开始训练
- **新手友好**: 所有 class 和关键逻辑均有中文注释

## 项目结构

```
captcha_recognizer/
├── config.yaml              # 配置文件 (模型/数据/训练/增强)
├── train.py                 # 训练入口
├── predict.py               # 推理脚本
├── export_onnx.py           # ONNX 导出
├── model/
│   ├── backbone.py          # 骨干网络 (ResNet 风格 CNN)
│   ├── head.py              # 识别头 (CTCHead / AttentionHead)
│   └── recognizer.py        # 完整模型 (组合 backbone + head)
├── data/
│   ├── dataset.py           # PyTorch Dataset + 批次整理
│   ├── augment.py           # 数据增强 (Albumentations)
│   └── tokenizer.py         # 字符编解码器
└── tools/
    └── generate_captcha.py  # 验证码生成工具
```

## 模型架构

```
输入图片 [B, 1, 64, 160]
         │
         ▼
┌─────────────────────────┐
│   ResNet Backbone        │  卷积 + 残差连接 + 池化
│   提取图像视觉特征        │  输出: [B, 40, 256] 特征序列
└────────────┬────────────┘
             │
     ┌───────┴───────┐
     ▼               ▼
┌─────────┐   ┌─────────────┐
│ CTC Head │   │ Attention   │
│ (简单)   │   │ Head (强大)  │
│ Linear   │   │ Transformer │
│ + CTC    │   │ Decoder     │
│   Loss   │   │ + CE Loss   │
└─────────┘   └─────────────┘
     │               │
     ▼               ▼
  预测结果          预测结果
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成训练数据

```bash
python tools/generate_captcha.py --num_train 20000 --num_val 2000
```

生成的验证码示例:

| 图片 | 标签 |
|------|------|
| ![captcha](https://via.placeholder.com/160x64?text=a3kp) | a3kp |

### 3. 开始训练

```bash
python train.py
```

训练过程:

```
使用设备: cuda
字符集大小: 36, 词表大小: 39
训练集: 20000 张, 验证集: 2000 张
模型参数量: 1,234,567

Epoch 1/100: 100%|████████| 156/156 [00:12] loss=12.3456 lr=0.000100
Epoch 1 | 平均损失: 15.2341 | 验证准确率: 0.0010 | 学习率: 0.000100
  ↑ 最优模型已保存 (准确率: 0.0010)

Epoch 10/100: 100%|████████| 156/156 [00:11] loss=0.8234 lr=0.000950
Epoch 10 | 平均损失: 1.0234 | 验证准确率: 0.6500 | 学习率: 0.000950
  ↑ 最优模型已保存 (准确率: 0.6500)
...
```

### 4. 推理

```bash
# 识别单张图片
python predict.py test.png

# 识别整个文件夹 (并计算准确率)
python predict.py data/val/
```

### 5. 导出 ONNX

```bash
python export_onnx.py --model checkpoints/best.pth --output model.onnx
```

## 配置说明

编辑 `config.yaml` 自定义训练参数:

```yaml
model:
  backbone: "resnet"         # 骨干网络
  head: "ctc"                # "ctc" (推荐) 或 "attention"
  feature_dim: 256           # 特征维度

data:
  charset: "0123456789abcdefghijklmnopqrstuvwxyz"
  image_height: 64
  image_width: 160
  image_channel: 1           # 1=灰度, 3=彩色
  max_label_length: 10
  train_dir: "data/train"
  val_dir: "data/val"

train:
  epochs: 100
  batch_size: 128
  lr: 0.001                  # 学习率
  patience: 15               # 早停耐心值
  use_amp: true              # 混合精度
```

## CTC vs Attention

| 特性 | CTC | Attention |
|------|-----|-----------|
| 结构复杂度 | 低 (一个全连接层) | 高 (Transformer Decoder) |
| 训练速度 | 快 | 较慢 |
| 是否需要对齐标签 | 不需要 | 不需要 |
| 字符间依赖建模 | 不支持 | 支持 |
| 数据需求 | 较少 | 较多 |
| 推荐场景 | 大多数验证码 | 复杂文字识别 |

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 框架 | PyTorch | 动态图，调试方便 |
| 骨干网络 | ResNet (残差网络) | 残差连接解决梯度消失 |
| 激活函数 | GELU | 比 ReLU 更平滑 |
| 识别头 | CTC / Transformer | 两种模式可选 |
| 优化器 | AdamW | 修正的权重衰减 |
| 学习率 | OneCycleLR | 先升后降，收敛更快 |
| 数据增强 | Albumentations | GPU 友好，速度快 |
| 混合精度 | PyTorch AMP | FP16 加速 + 省显存 |
| 部署格式 | ONNX | 跨平台通用格式 |

## License

[MIT](LICENSE)
