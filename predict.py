"""
推理脚本

加载训练好的模型，对单张或多张图片进行识别。

用法:
  python predict.py image.png                          # 识别单张图片
  python predict.py image_dir/                         # 识别整个文件夹
  python predict.py image.png --model best.pth         # 指定模型
  python predict.py image.png --config my.yaml         # 指定配置
"""

import os
import argparse
import cv2
import torch
import numpy as np
import yaml

from data.tokenizer import Tokenizer
from model.recognizer import CaptchaRecognizer


def load_model(checkpoint_path: str, config_path: str = None, device: str = "cpu"):
    """
    加载训练好的模型

    Args:
        checkpoint_path: 模型文件路径 (.pth)
        config_path:     配置文件路径 (如果不提供，从 checkpoint 中读取)
        device:          运行设备

    Returns:
        (model, tokenizer, config) 三元组
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 优先使用外部配置文件，否则从 checkpoint 中读取
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint["config"]

    tokenizer = Tokenizer(config["data"]["charset"])
    model = CaptchaRecognizer(config, tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, config


def preprocess_image(image_path: str, config: dict) -> torch.Tensor:
    """
    预处理单张图片

    Args:
        image_path: 图片路径
        config:     配置字典

    Returns:
        [1, C, H, W] 的张量 (已添加 batch 维度)
    """
    data_config = config["data"]

    if data_config["image_channel"] == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        raise IOError(f"无法读取图片: {image_path}")

    image = cv2.resize(image, (data_config["image_width"], data_config["image_height"]))
    image = image.astype(np.float32) / 255.0

    if data_config["image_channel"] == 1:
        tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)    # [1, 1, H, W]
    else:
        tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    return tensor


def predict_single(model, tokenizer, config, image_path, device):
    """识别单张图片"""
    image = preprocess_image(image_path, config).to(device)
    pred_indices = model.predict(image)
    head_type = config["model"]["head"]
    return tokenizer.decode(pred_indices[0], mode=head_type)


def main():
    parser = argparse.ArgumentParser(description="验证码识别推理")
    parser.add_argument("input", type=str, help="图片路径或图片文件夹")
    parser.add_argument("--model", type=str, default="checkpoints/best.pth", help="模型路径")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--device", type=str, default="auto", help="设备: auto/cpu/cuda")
    args = parser.parse_args()

    # 选择设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # 加载模型
    model, tokenizer, config = load_model(args.model, args.config, device)
    print(f"模型已加载 (设备: {device})")

    # 单张图片 or 文件夹
    if os.path.isfile(args.input):
        result = predict_single(model, tokenizer, config, args.input, device)
        print(f"{args.input} → {result}")

    elif os.path.isdir(args.input):
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        files = sorted(
            f for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in extensions
        )

        correct = 0
        total = 0
        for filename in files:
            filepath = os.path.join(args.input, filename)
            result = predict_single(model, tokenizer, config, filepath, device)

            # 如果文件名包含标签，可以计算准确率
            gt_label = os.path.splitext(filename)[0].split("_")[0].lower()
            match = "✓" if result == gt_label else "✗"
            if result == gt_label:
                correct += 1
            total += 1

            print(f"  {match}  {filename} → 预测: {result}, 真实: {gt_label}")

        if total > 0:
            print(f"\n准确率: {correct}/{total} = {correct / total:.4f}")
    else:
        print(f"路径不存在: {args.input}")


if __name__ == "__main__":
    main()
