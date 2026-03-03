"""
ONNX 导出页面

将训练好的 PyTorch 模型导出为 ONNX 格式, 对应 export_onnx.py。
支持指定模型路径、输出路径, 导出后自动验证。
"""

import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QFileDialog,
)
from qfluentwidgets import (
    ScrollArea, TitleLabel, SubtitleLabel, BodyLabel,
    LineEdit, PrimaryPushButton, PushButton, CardWidget,
    InfoBar, InfoBarPosition, ProgressBar,
    FluentIcon as FIF,
)

from gui.worker import WorkerThread


class ExportPage(ScrollArea):
    """ONNX 导出页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("export_page")
        self.worker = None

        self._init_ui()

    def _init_ui(self):
        self.scroll_widget = QWidget()
        self.setWidget(self.scroll_widget)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.scroll_widget.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self.scroll_widget)
        layout.setContentsMargins(36, 20, 36, 20)
        layout.setSpacing(16)

        # ── 标题 ──
        layout.addWidget(TitleLabel("ONNX 导出"))
        layout.addWidget(BodyLabel(
            "将 PyTorch 模型导出为 ONNX 格式, 可被 ONNX Runtime / TensorRT / OpenVINO 加载推理"
        ))

        # ── 路径配置 ──
        path_card = CardWidget(self.scroll_widget)
        path_layout = QVBoxLayout(path_card)
        path_layout.setContentsMargins(20, 16, 20, 16)
        path_layout.addWidget(SubtitleLabel("路径配置"))

        # 模型输入路径
        model_col = QVBoxLayout()
        model_col.addWidget(BodyLabel("PyTorch 模型文件 (.pth)"))
        model_row = QHBoxLayout()
        self.model_path_edit = LineEdit()
        self.model_path_edit.setText("checkpoints/best.pth")
        model_row.addWidget(self.model_path_edit)
        self.browse_model_btn = PushButton("浏览")
        self.browse_model_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self.browse_model_btn)
        model_col.addLayout(model_row)
        path_layout.addLayout(model_col)

        # ONNX 输出路径
        output_col = QVBoxLayout()
        output_col.addWidget(BodyLabel("ONNX 输出路径"))
        output_row = QHBoxLayout()
        self.output_path_edit = LineEdit()
        self.output_path_edit.setText("checkpoints/model.onnx")
        output_row.addWidget(self.output_path_edit)
        self.browse_output_btn = PushButton("浏览")
        self.browse_output_btn.clicked.connect(self._browse_output)
        output_row.addWidget(self.browse_output_btn)
        output_col.addLayout(output_row)
        path_layout.addLayout(output_col)

        layout.addWidget(path_card)

        # ── 操作 ──
        action_card = CardWidget(self.scroll_widget)
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(20, 16, 20, 16)

        self.progress_bar = ProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.export_btn = PrimaryPushButton(FIF.SAVE, "导出 ONNX")
        self.export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self.export_btn)
        action_layout.addLayout(btn_layout)

        layout.addWidget(action_card)

        # ── 日志 ──
        log_card = CardWidget(self.scroll_widget)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(20, 16, 20, 16)
        log_layout.addWidget(SubtitleLabel("日志"))

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #cccccc;"
            " font-family: Consolas, 'Courier New', monospace; font-size: 13px;"
            " border-radius: 8px; padding: 8px; }"
        )
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_card)

        layout.addStretch()

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "checkpoints", "PyTorch 模型 (*.pth);;所有文件 (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "选择输出路径", "checkpoints/model.onnx", "ONNX 模型 (*.onnx);;所有文件 (*)"
        )
        if path:
            self.output_path_edit.setText(path)

    def _on_export(self):
        if self.worker and self.worker.isRunning():
            InfoBar.warning("提示", "导出任务正在运行中", parent=self,
                            position=InfoBarPosition.TOP, duration=2000)
            return

        model_path = self.model_path_edit.text().strip()
        output_path = self.output_path_edit.text().strip()

        if not model_path or not os.path.exists(model_path):
            InfoBar.error("路径错误", "模型文件不存在", parent=self,
                          position=InfoBarPosition.TOP, duration=3000)
            return

        if not output_path:
            InfoBar.error("路径错误", "请指定输出路径", parent=self,
                          position=InfoBarPosition.TOP, duration=3000)
            return

        self.log_text.clear()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.export_btn.setEnabled(False)

        self.worker = WorkerThread(
            target=self._run_export,
            args=(model_path, output_path),
        )
        self.worker.log_signal.connect(self._append_log)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    @staticmethod
    def _run_export(model_path, output_path):
        from export_onnx import export
        export(model_path, output_path)

    def _append_log(self, text):
        self.log_text.appendPlainText(text)

    def _on_finished(self, success, msg):
        self.export_btn.setEnabled(True)
        self.progress_bar.setValue(100 if success else 0)
        if success:
            InfoBar.success("完成", "ONNX 模型导出成功", parent=self,
                            position=InfoBarPosition.TOP, duration=3000)
        else:
            InfoBar.error("失败", msg, parent=self,
                          position=InfoBarPosition.TOP, duration=5000)
