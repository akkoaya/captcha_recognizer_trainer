<h1 align="center">Captcha Recognizer Trainer</h1>

<p align="center">
  <b>基于 PyTorch 的现代验证码识别框架</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/ONNX-Supported-purple.svg" alt="ONNX Supported">
  <img src="https://img.shields.io/badge/GUI-QFluentWidgets-teal.svg" alt="GUI: QFluentWidgets">
</p>

<p align="center">
  采用 ResNet + CTC/Attention 架构，支持固定长度与不定长验证码、混合精度训练、ONNX 导出、<br>
  QFluentWidgets 图形界面、丰富的数据增强。代码注释详尽，适合深度学习初学者阅读和学习。
</p>

---

## 目录

- [特性](#特性)
- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [环境要求](#环境要求)
  - [安装](#安装)
  - [GUI 启动](#gui-启动)
  - [生成训练数据](#生成训练数据)
  - [训练](#训练)
  - [推理](#推理)
  - [导出 ONNX](#导出-onnx)
- [不定长验证码](#不定长验证码)
- [配置说明](#配置说明)
- [CTC vs Attention](#ctc-vs-attention)
- [技术栈](#技术栈)
- [Contributing](#contributing)
- [License](#license)

## 特性

- **现代架构** — ResNet 残差骨干网络 + CTC / Transformer Attention 双模式识别头
- **不定长支持** — 同时支持固定长度和不定长验证码，训练时按长度分组统计准确率
- **图形界面** — 基于 [QFluentWidgets](https://qfluentwidgets.com) (Fluent Design) 的 GUI，覆盖数据生成 / 训练 / 推理 / 导出全流程
- **训练优化** — AdamW 优化器、OneCycleLR 学习率调度、混合精度 (AMP)、早停、梯度裁剪
- **数据增强** — 基于 [Albumentations](https://github.com/albumentations-team/albumentations)，高效且可组合
- **一键部署** — 导出 ONNX 格式，支持 ONNX Runtime / TensorRT 推理
- **开箱即用** — 内置验证码生成工具，无需额外数据即可开始训练
- **新手友好** — 所有 class 和关键逻辑均有中文注释

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

## 项目结构

```
captcha_recognizer_trainer/
├── config.yaml              # 配置文件 (模型/数据/训练/增强)
├── train.py                 # 训练入口
├── predict.py               # 推理脚本
├── export_onnx.py           # ONNX 导出
├── run_gui.py               # GUI 启动入口
├── model/
│   ├── backbone.py          # 骨干网络 (ResNet 风格 CNN)
│   ├── head.py              # 识别头 (CTCHead / AttentionHead)
│   └── recognizer.py        # 完整模型 (组合 backbone + head)
├── data/
│   ├── dataset.py           # PyTorch Dataset + 批次整理
│   ├── augment.py           # 数据增强 (Albumentations)
│   └── tokenizer.py         # 字符编解码器
├── gui/                     # QFluentWidgets 图形界面
│   ├── main_window.py       # 主窗口 (FluentWindow 侧边栏导航)
│   ├── worker.py            # 后台工作线程
│   └── pages/
│       ├── generate_page.py # 数据生成页
│       ├── train_page.py    # 模型训练页
│       ├── predict_page.py  # 推理预测页
│       └── export_page.py   # ONNX 导出页
└── tools/
    └── generate_captcha.py  # 验证码生成工具
```

## 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA (可选，推荐用于加速训练)

### 安装

```bash
git clone https://github.com/akkoaya/captcha_recognizer_trainer.git
cd captcha_recognizer_trainer
pip install -r requirements.txt
```

### GUI 启动

提供完整的图形界面，覆盖数据生成、训练、推理、导出全部功能:

```bash
python run_gui.py
```

GUI 包含四个功能页面:

| 页面 | 功能 |
|------|------|
| 数据生成 | 配置字符集、长度范围、图片尺寸，一键生成训练/验证集 |
| 模型训练 | 完整参数编辑器，实时训练日志，启动/停止控制 |
| 推理预测 | 选择图片或文件夹批量识别，结果表格 + 准确率统计 |
| ONNX 导出 | 选择模型一键导出 ONNX，自动验证 |

### 生成训练数据

```bash
# 默认生成 4~6 字符不定长验证码
python tools/generate_captcha.py --num_train 20000 --num_val 2000

# 固定 4 字符
python tools/generate_captcha.py --min_length 4 --max_length 4

# 1~8 字符不定长
python tools/generate_captcha.py --min_length 1 --max_length 8
```

### 训练

```bash
python train.py
```

训练过程示例输出:

```
使用设备: cuda
字符集大小: 36, 词表大小: 39
训练集: 20000 张, 验证集: 2000 张
标签长度分布 (不定长模式):
  训练集: 4字符=6634张  5字符=6688张  6字符=6678张
  验证集: 4字符=662张  5字符=681张  6字符=657张
模型参数量: 1,234,567

Epoch 1/100: 100%|████████| 156/156 [00:12] loss=12.3456 lr=0.000100
Epoch 1 | 平均损失: 15.2341 | 验证准确率: 0.0010 | 学习率: 0.000100
  各长度准确率: 4字符=0.002(1/662), 5字符=0.000(0/681), 6字符=0.000(0/657)
  ↑ 最优模型已保存 (准确率: 0.0010)
...
```

### 推理

```bash
# 识别单张图片
python predict.py test.png

# 识别整个文件夹 (并计算准确率)
python predict.py data/val/
```

### 导出 ONNX

```bash
python export_onnx.py --model checkpoints/best.pth --output model.onnx
```

## 不定长验证码

框架同时支持固定长度和不定长验证码训练。CTC 损失函数天然支持不定长序列对齐，Attention 通过 SOS/EOS 标记实现自回归生成，因此模型层无需额外修改。

通过 `config.yaml` 中的 `min_label_length` 和 `max_label_length` 控制:

```yaml
data:
  min_label_length: 4    # 最少 4 个字符
  max_label_length: 6    # 最多 6 个字符
  # 若 min = max, 则为固定长度模式
```

训练时会自动检测数据集中的标签长度分布，并在每个 epoch 的验证阶段按长度分组报告准确率:

```
标签长度分布 (不定长模式):
  训练集: 4字符=6634张  5字符=6688张  6字符=6678张
  验证集: 4字符=662张  5字符=681张  6字符=657张
...
  各长度准确率: 4字符=0.952(630/662), 5字符=0.934(636/681), 6字符=0.892(586/657)
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
  min_label_length: 4        # 最小标签长度
  max_label_length: 10       # 最大标签长度
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
| GUI | PyQt6 + QFluentWidgets | Fluent Design 风格图形界面 |
| 骨干网络 | ResNet (残差网络) | 残差连接解决梯度消失 |
| 激活函数 | GELU | 比 ReLU 更平滑 |
| 识别头 | CTC / Transformer | 两种模式可选 |
| 优化器 | AdamW | 修正的权重衰减 |
| 学习率 | OneCycleLR | 先升后降，收敛更快 |
| 数据增强 | Albumentations | GPU 友好，速度快 |
| 混合精度 | PyTorch AMP | FP16 加速 + 省显存 |
| 部署格式 | ONNX | 跨平台通用格式 |

## Contributing

欢迎贡献代码! 请遵循以下步骤:

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

如果你发现了 Bug 或有新功能建议，欢迎提交 [Issue](../../issues)。

## License

本项目基于 [MIT License](LICENSE) 开源。
