"""
数据增强模块

使用 Albumentations 库实现高效的图像增强。
训练时随机施加增强以提升模型泛化能力，验证/推理时不做增强。
"""

import albumentations as A


def get_train_transforms(config: dict) -> A.Compose:
    """
    构建训练时的数据增强管线

    所有增强都有概率控制 (p 参数)，不会每次都执行。
    增强顺序也经过考量：先做几何变换，再做像素变换。

    Args:
        config: 增强配置字典

    Returns:
        Albumentations 变换组合
    """
    aug_config = config.get("augment", {})
    transforms = []

    # ── 几何变换 (改变形状/位置) ──

    # 随机旋转：模拟验证码倾斜
    rotate_limit = aug_config.get("rotate_limit", 10)
    if rotate_limit > 0:
        transforms.append(
            A.Rotate(
                limit=rotate_limit,
                border_mode=0,           # 用常数填充边界
                value=255,               # 白色填充
                p=0.5,
            )
        )

    # 透视变形：模拟拍摄角度变化
    if aug_config.get("perspective", False):
        transforms.append(
            A.Perspective(
                scale=(0.03, 0.08),      # 变形幅度
                pad_mode=0,
                pad_val=255,
                p=0.3,
            )
        )

    # ── 模糊类 (降低清晰度) ──

    blur_limit = aug_config.get("blur_limit", 3)
    if blur_limit > 1:
        transforms.append(
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, blur_limit), p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.3,
            )
        )

    # ── 噪声 ──

    if aug_config.get("noise", False):
        transforms.append(
            A.GaussNoise(
                std_range=(5.0 / 255, 25.0 / 255),
                p=0.3,
            )
        )

    # ── 亮度/对比度 ──

    if aug_config.get("brightness_contrast", False):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4,
            )
        )

    return A.Compose(transforms)


def get_val_transforms() -> A.Compose:
    """
    验证/推理时的变换 (不做任何增强，仅保持格式一致)

    Returns:
        空的 Albumentations 变换组合
    """
    return A.Compose([])
