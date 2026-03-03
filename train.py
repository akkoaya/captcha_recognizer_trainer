"""
训练脚本

完整的训练流程:
  1. 加载配置
  2. 准备数据集和数据加载器
  3. 构建模型
  4. 训练循环 (含验证、早停、学习率调度、混合精度)
  5. 保存最优模型

用法:
  python train.py                     # 使用默认配置
  python train.py --config my.yaml    # 指定配置文件
"""

import os
import argparse
import yaml
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.tokenizer import Tokenizer
from data.dataset import CaptchaDataset, ctc_collate_fn, attention_collate_fn
from data.augment import get_train_transforms, get_val_transforms
from model.recognizer import CaptchaRecognizer


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate(model, val_loader, tokenizer, device, head_type):
    """
    在验证集上计算准确率

    使用整体准确率: 整个验证码完全匹配才算对。
    例如真实标签 "a3kp"，预测 "a3kq"，算作错误。

    同时统计各长度的准确率，便于分析不定长场景下的性能。
    """
    model.eval()
    correct = 0
    total = 0

    # 按长度分组统计
    length_correct = defaultdict(int)
    length_total = defaultdict(int)

    with torch.no_grad():
        for images, targets, target_lengths, texts in val_loader:
            images = images.to(device)
            pred_indices = model.predict(images)

            for pred, gt_text in zip(pred_indices, texts):
                pred_text = tokenizer.decode(pred, mode=head_type)
                gt_len = len(gt_text)
                length_total[gt_len] += 1
                if pred_text == gt_text:
                    correct += 1
                    length_correct[gt_len] += 1
                total += 1

    accuracy = correct / total if total > 0 else 0

    # 构建各长度准确率字典
    per_length_accuracy = {}
    for length in sorted(length_total.keys()):
        acc = length_correct[length] / length_total[length]
        per_length_accuracy[length] = (acc, length_correct[length], length_total[length])

    return accuracy, per_length_accuracy


def train(config_path: str):
    """主训练函数"""

    # ═══════════════ 1. 加载配置 ═══════════════
    config = load_config(config_path)
    train_config = config["train"]
    data_config = config["data"]
    head_type = config["model"]["head"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ═══════════════ 2. 准备字符编解码器 ═══════════════
    tokenizer = Tokenizer(data_config["charset"])
    print(f"字符集大小: {len(tokenizer.charset)}, 词表大小: {tokenizer.vocab_size}")

    # ═══════════════ 3. 准备数据集 ═══════════════
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms()

    train_dataset = CaptchaDataset(
        data_dir=data_config["train_dir"],
        tokenizer=tokenizer,
        img_height=data_config["image_height"],
        img_width=data_config["image_width"],
        img_channel=data_config["image_channel"],
        transform=train_transform,
    )
    val_dataset = CaptchaDataset(
        data_dir=data_config["val_dir"],
        tokenizer=tokenizer,
        img_height=data_config["image_height"],
        img_width=data_config["image_width"],
        img_channel=data_config["image_channel"],
        transform=val_transform,
    )

    # 根据识别头类型选择批次整理函数
    collate_fn = ctc_collate_fn if head_type == "ctc" else attention_collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,                           # 训练时打乱顺序
        num_workers=train_config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,                        # 加速 GPU 传输
        drop_last=True,                         # 丢弃不完整的最后一个 batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,                          # 验证时不打乱
        num_workers=train_config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

    # 统计标签长度分布
    train_length_dist = defaultdict(int)
    for _, label_text in train_dataset.samples:
        train_length_dist[len(label_text)] += 1
    val_length_dist = defaultdict(int)
    for _, label_text in val_dataset.samples:
        val_length_dist[len(label_text)] += 1

    is_variable_length = len(train_length_dist) > 1
    if is_variable_length:
        print("标签长度分布 (不定长模式):")
        print(f"  训练集: ", end="")
        for length in sorted(train_length_dist.keys()):
            count = train_length_dist[length]
            print(f"{length}字符={count}张", end="  ")
        print()
        print(f"  验证集: ", end="")
        for length in sorted(val_length_dist.keys()):
            count = val_length_dist[length]
            print(f"{length}字符={count}张", end="  ")
        print()
    else:
        fixed_len = list(train_length_dist.keys())[0]
        print(f"标签长度: 固定 {fixed_len} 字符")

    # ═══════════════ 4. 构建模型 ═══════════════
    model = CaptchaRecognizer(config, tokenizer.vocab_size).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # ═══════════════ 5. 优化器 + 学习率调度 ═══════════════

    # AdamW: 修正了 Adam 的权重衰减问题，当前最主流的优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
    )

    # OneCycleLR: 学习率先升后降，比固定衰减收敛更快
    #   热身阶段: 学习率从很小增加到 max_lr
    #   退火阶段: 学习率从 max_lr 逐渐降低到接近 0
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_config["lr"],
        epochs=train_config["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,          # 前 10% 的步数用于热身
    )

    # ═══════════════ 6. 混合精度训练 ═══════════════
    #   FP16 计算更快、省显存；关键操作仍用 FP32 保证精度
    use_amp = train_config.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ═══════════════ 7. 训练循环 ═══════════════
    save_dir = train_config.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    best_accuracy = 0.0
    patience_counter = 0
    patience = train_config.get("patience", 15)
    grad_clip = train_config.get("grad_clip", 5.0)

    for epoch in range(1, train_config["epochs"] + 1):

        # ── 训练阶段 ──
        model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{train_config['epochs']}",
            leave=True,
        )

        for images, targets, target_lengths, texts in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            # 前向传播 (混合精度)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = model(images, targets, target_lengths)

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # 梯度裁剪: 防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # 更新权重
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.6f}",
            )

        avg_loss = total_loss / num_batches

        # ── 验证阶段 ──
        accuracy, per_length_accuracy = validate(model, val_loader, tokenizer, device, head_type)

        print(
            f"Epoch {epoch} | "
            f"平均损失: {avg_loss:.4f} | "
            f"验证准确率: {accuracy:.4f} | "
            f"学习率: {scheduler.get_last_lr()[0]:.6f}"
        )

        # 不定长模式下显示各长度的准确率
        if len(per_length_accuracy) > 1:
            parts = []
            for length, (acc, correct_cnt, total_cnt) in per_length_accuracy.items():
                parts.append(f"{length}字符={acc:.3f}({correct_cnt}/{total_cnt})")
            print(f"  各长度准确率: {', '.join(parts)}")

        # ── 保存最优模型 ──
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            save_path = os.path.join(save_dir, "best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "accuracy": accuracy,
                    "config": config,
                },
                save_path,
            )
            print(f"  ↑ 最优模型已保存 (准确率: {accuracy:.4f})")
        else:
            patience_counter += 1
            print(f"  准确率未提升 ({patience_counter}/{patience})")

        # ── 早停 (Early Stopping) ──
        #   连续多个 epoch 准确率不提升时，停止训练防止过拟合
        if patience_counter >= patience:
            print(f"\n连续 {patience} 个 epoch 准确率未提升，提前停止训练。")
            break

    print(f"\n训练完成！最优验证准确率: {best_accuracy:.4f}")
    print(f"模型已保存至: {os.path.join(save_dir, 'best.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练验证码识别模型")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    train(args.config)
