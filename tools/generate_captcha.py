"""
验证码生成工具

使用 captcha 库生成训练和验证数据集。
生成的图片文件名格式: 标签_序号.png，如 a3kp_000001.png

用法:
  python tools/generate_captcha.py                              # 使用默认参数
  python tools/generate_captcha.py --num_train 50000 --num_val 5000
  python tools/generate_captcha.py --charset "0123456789" --length 4
"""

import os
import random
import string
import argparse

from captcha.image import ImageCaptcha


def generate_dataset(
    output_dir: str,
    num_samples: int,
    charset: str,
    min_length: int,
    max_length: int,
    width: int,
    height: int,
):
    """
    生成验证码数据集

    Args:
        output_dir:  输出文件夹
        num_samples: 生成数量
        charset:     字符集
        min_length:  最小字符数
        max_length:  最大字符数
        width:       图片宽度
        height:      图片高度
    """
    os.makedirs(output_dir, exist_ok=True)

    generator = ImageCaptcha(width=width, height=height)

    print(f"正在生成 {num_samples} 张验证码到 {output_dir}/ ...")

    for i in range(num_samples):
        # 随机生成标签
        length = random.randint(min_length, max_length)
        text = "".join(random.choices(charset, k=length))

        # 生成图片并保存
        # 文件名格式: 标签_序号.png
        filename = f"{text}_{i:06d}.png"
        filepath = os.path.join(output_dir, filename)
        generator.write(text, filepath)

        if (i + 1) % 1000 == 0:
            print(f"  已生成 {i + 1}/{num_samples}")

    print(f"完成! 共生成 {num_samples} 张验证码。")


def main():
    parser = argparse.ArgumentParser(description="生成验证码训练数据")
    parser.add_argument("--train_dir", type=str, default="data/train", help="训练集输出目录")
    parser.add_argument("--val_dir", type=str, default="data/val", help="验证集输出目录")
    parser.add_argument("--num_train", type=int, default=20000, help="训练集数量")
    parser.add_argument("--num_val", type=int, default=2000, help="验证集数量")
    parser.add_argument(
        "--charset",
        type=str,
        default="0123456789abcdefghijklmnopqrstuvwxyz",
        help="字符集",
    )
    parser.add_argument("--min_length", type=int, default=4, help="最小字符数")
    parser.add_argument("--max_length", type=int, default=4, help="最大字符数")
    parser.add_argument("--width", type=int, default=160, help="图片宽度")
    parser.add_argument("--height", type=int, default=64, help="图片高度")
    args = parser.parse_args()

    # 生成训练集
    generate_dataset(
        output_dir=args.train_dir,
        num_samples=args.num_train,
        charset=args.charset,
        min_length=args.min_length,
        max_length=args.max_length,
        width=args.width,
        height=args.height,
    )

    # 生成验证集
    generate_dataset(
        output_dir=args.val_dir,
        num_samples=args.num_val,
        charset=args.charset,
        min_length=args.min_length,
        max_length=args.max_length,
        width=args.width,
        height=args.height,
    )

    print("\n数据集生成完毕!")
    print(f"  训练集: {args.train_dir}/ ({args.num_train} 张)")
    print(f"  验证集: {args.val_dir}/ ({args.num_val} 张)")
    print(f"\n下一步: python train.py")


if __name__ == "__main__":
    main()
