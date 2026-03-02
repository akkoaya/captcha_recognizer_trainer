"""
ONNX 模型导出

将训练好的 PyTorch 模型导出为 ONNX 格式，便于跨平台部署。

ONNX (Open Neural Network Exchange) 是通用的模型交换格式，
可以被 ONNX Runtime、TensorRT、OpenVINO 等推理引擎加载。

用法:
  python export_onnx.py                                     # 使用默认路径
  python export_onnx.py --model best.pth --output model.onnx
"""

import argparse
import torch
import yaml

from data.tokenizer import Tokenizer
from model.recognizer import CaptchaRecognizer


class CaptchaRecognizerExport(torch.nn.Module):
    """
    导出专用的模型包装器

    推理时只需要 backbone + head 的前向传播 + argmax 解码，
    不需要损失计算相关的逻辑。
    """

    def __init__(self, model: CaptchaRecognizer):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head
        self.head_type = model.head_type

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        logits = self.head(features)
        if self.head_type == "ctc":
            return logits.argmax(dim=2)    # [B, T] 贪心解码
        else:
            return logits.argmax(dim=2)    # 简化: attention 也可以用 argmax


def export(model_path: str, output_path: str, config_path: str = None):
    """导出模型为 ONNX 格式"""

    # 加载 checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    data_config = config["data"]

    # 重建模型
    tokenizer = Tokenizer(data_config["charset"])
    model = CaptchaRecognizer(config, tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 包装为导出模型
    export_model = CaptchaRecognizerExport(model)
    export_model.eval()

    # 构造示例输入
    dummy_input = torch.randn(
        1,
        data_config["image_channel"],
        data_config["image_height"],
        data_config["image_width"],
    )

    # 导出 ONNX
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["prediction"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "prediction": {0: "batch_size"},
        },
        opset_version=17,
    )

    print(f"ONNX 模型已导出至: {output_path}")

    # 验证导出的模型
    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证通过")
    except ImportError:
        print("提示: 安装 onnx 包可以验证导出的模型 (pip install onnx)")

    # 用 ONNX Runtime 测试推理
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(output_path)
        test_input = np.random.randn(
            1,
            data_config["image_channel"],
            data_config["image_height"],
            data_config["image_width"],
        ).astype(np.float32)

        result = session.run(None, {"image": test_input})
        print(f"ONNX Runtime 推理测试通过, 输出形状: {result[0].shape}")
    except ImportError:
        print("提示: 安装 onnxruntime 包可以测试推理 (pip install onnxruntime)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出 ONNX 模型")
    parser.add_argument("--model", type=str, default="checkpoints/best.pth", help="模型路径")
    parser.add_argument("--output", type=str, default="checkpoints/model.onnx", help="输出路径")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()
    export(args.model, args.output, args.config)
