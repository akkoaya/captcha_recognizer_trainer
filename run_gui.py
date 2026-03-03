"""
验证码识别训练器 - GUI 启动入口

使用 QFluentWidgets 构建的图形界面, 整合数据生成、模型训练、推理预测、ONNX 导出功能。

用法:
  python run_gui.py
"""

import sys
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import setTheme, Theme

from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    # 跟随系统主题
    setTheme(Theme.AUTO)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
