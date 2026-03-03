"""
主窗口

使用 QFluentWidgets 的 FluentWindow 构建侧边栏导航主窗口。
按功能分为四个页面: 数据生成、模型训练、推理预测、ONNX 导出。
"""

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication
from qfluentwidgets import (
    FluentWindow,
    FluentIcon as FIF,
    NavigationItemPosition,
    setTheme,
    Theme,
)

from gui.pages.generate_page import GeneratePage
from gui.pages.train_page import TrainPage
from gui.pages.predict_page import PredictPage
from gui.pages.export_page import ExportPage


class MainWindow(FluentWindow):
    """验证码识别训练器 - 主窗口"""

    def __init__(self):
        super().__init__()
        self._init_pages()
        self._init_window()

    def _init_pages(self):
        """初始化各功能页面并添加到导航栏"""
        self.generate_page = GeneratePage(self)
        self.train_page = TrainPage(self)
        self.predict_page = PredictPage(self)
        self.export_page = ExportPage(self)

        self.addSubInterface(self.generate_page, FIF.ADD, "数据生成")
        self.addSubInterface(self.train_page, FIF.PLAY, "模型训练")
        self.addSubInterface(self.predict_page, FIF.SEARCH, "推理预测")
        self.addSubInterface(
            self.export_page, FIF.SAVE, "ONNX 导出",
            position=NavigationItemPosition.BOTTOM,
        )

    def _init_window(self):
        """初始化窗口属性"""
        self.resize(1000, 720)
        self.setMinimumSize(800, 600)
        self.setWindowTitle("验证码识别训练器")

        # 居中显示
        desktop = QApplication.desktop()
        if desktop is not None:
            rect = desktop.availableGeometry()
            self.move(
                (rect.width() - self.width()) // 2,
                (rect.height() - self.height()) // 2,
            )
