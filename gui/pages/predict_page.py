"""
推理预测页面

加载训练好的模型, 对单张或多张图片进行验证码识别, 对应 predict.py。
支持拖拽图片、文件夹选择、实时显示识别结果。
"""

import os
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog,
    QPlainTextEdit, QLabel, QSizePolicy, QTableWidgetItem,
)
from qfluentwidgets import (
    ScrollArea, TitleLabel, SubtitleLabel, BodyLabel,
    LineEdit, PrimaryPushButton, PushButton, CardWidget,
    InfoBar, InfoBarPosition, ProgressBar, TableWidget,
    FluentIcon as FIF,
)

from gui.worker import WorkerThread


class PredictPage(ScrollArea):
    """推理预测页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("predict_page")
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
        layout.addWidget(TitleLabel("推理预测"))
        layout.addWidget(BodyLabel("加载训练好的模型, 对单张图片或整个文件夹进行验证码识别"))

        # ── 模型路径 ──
        model_card = CardWidget(self.scroll_widget)
        model_layout = QVBoxLayout(model_card)
        model_layout.setContentsMargins(20, 16, 20, 16)
        model_layout.addWidget(SubtitleLabel("模型设置"))

        row_model = QHBoxLayout()

        model_col = QVBoxLayout()
        model_col.addWidget(BodyLabel("模型文件 (.pth)"))
        model_row = QHBoxLayout()
        self.model_path_edit = LineEdit()
        self.model_path_edit.setText("checkpoints/best.pth")
        model_row.addWidget(self.model_path_edit)
        self.browse_model_btn = PushButton("浏览")
        self.browse_model_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self.browse_model_btn)
        model_col.addLayout(model_row)
        row_model.addLayout(model_col)

        model_layout.addLayout(row_model)
        layout.addWidget(model_card)

        # ── 输入 ──
        input_card = CardWidget(self.scroll_widget)
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(20, 16, 20, 16)
        input_layout.addWidget(SubtitleLabel("输入"))

        input_row = QHBoxLayout()
        self.input_path_edit = LineEdit()
        self.input_path_edit.setPlaceholderText("图片路径或图片文件夹路径")
        input_row.addWidget(self.input_path_edit)

        self.browse_img_btn = PushButton("选择图片")
        self.browse_img_btn.clicked.connect(self._browse_image)
        input_row.addWidget(self.browse_img_btn)

        self.browse_dir_btn = PushButton("选择文件夹")
        self.browse_dir_btn.clicked.connect(self._browse_dir)
        input_row.addWidget(self.browse_dir_btn)

        input_layout.addLayout(input_row)
        layout.addWidget(input_card)

        # ── 操作 ──
        action_card = CardWidget(self.scroll_widget)
        action_layout_card = QVBoxLayout(action_card)
        action_layout_card.setContentsMargins(20, 16, 20, 16)

        self.progress_bar = ProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        action_layout_card.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.predict_btn = PrimaryPushButton(FIF.SEARCH, "开始识别")
        self.predict_btn.clicked.connect(self._on_predict)
        btn_layout.addWidget(self.predict_btn)
        action_layout_card.addLayout(btn_layout)

        layout.addWidget(action_card)

        # ── 单张图片预览 ──
        preview_card = CardWidget(self.scroll_widget)
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(20, 16, 20, 16)
        preview_layout.addWidget(SubtitleLabel("预览"))

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(80)
        self.preview_label.setText("加载图片后显示预览")
        self.preview_label.setStyleSheet("color: #888;")
        preview_layout.addWidget(self.preview_label)

        self.result_label = BodyLabel("")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.result_label)

        layout.addWidget(preview_card)

        # ── 批量结果表格 ──
        result_card = CardWidget(self.scroll_widget)
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(20, 16, 20, 16)
        result_layout.addWidget(SubtitleLabel("批量识别结果"))

        self.result_table = TableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["文件名", "预测结果", "真实标签", "匹配"])
        self.result_table.setMinimumHeight(200)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        result_layout.addWidget(self.result_table)

        self.accuracy_label = BodyLabel("")
        result_layout.addWidget(self.accuracy_label)

        layout.addWidget(result_card)

        # ── 日志 ──
        log_card = CardWidget(self.scroll_widget)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(20, 16, 20, 16)
        log_layout.addWidget(SubtitleLabel("日志"))

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
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

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)"
        )
        if path:
            self.input_path_edit.setText(path)

    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.input_path_edit.setText(path)

    def _on_predict(self):
        if self.worker and self.worker.isRunning():
            InfoBar.warning("提示", "识别任务正在运行中", parent=self,
                            position=InfoBarPosition.TOP, duration=2000)
            return

        model_path = self.model_path_edit.text().strip()
        input_path = self.input_path_edit.text().strip()

        if not model_path or not os.path.exists(model_path):
            InfoBar.error("路径错误", "模型文件不存在", parent=self,
                          position=InfoBarPosition.TOP, duration=3000)
            return

        if not input_path or not os.path.exists(input_path):
            InfoBar.error("路径错误", "输入路径不存在", parent=self,
                          position=InfoBarPosition.TOP, duration=3000)
            return

        self.log_text.clear()
        self.result_table.setRowCount(0)
        self.accuracy_label.setText("")
        self.result_label.setText("")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.predict_btn.setEnabled(False)

        # 单张图片预览
        if os.path.isfile(input_path):
            pixmap = QPixmap(input_path)
            if not pixmap.isNull():
                self.preview_label.setPixmap(
                    pixmap.scaled(320, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                )

        self.worker = WorkerThread(
            target=self._run_predict,
            args=(model_path, input_path, self),
        )
        self.worker.log_signal.connect(self._append_log)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    @staticmethod
    def _run_predict(model_path, input_path, page):
        """在子线程中执行推理 (通过信号更新 UI)"""
        import torch
        from predict import load_model, predict_single

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, config = load_model(model_path, device=device)
        print(f"模型已加载 (设备: {device})")

        if os.path.isfile(input_path):
            result = predict_single(model, tokenizer, config, input_path, device)
            print(f"识别结果: {result}")
            # 通过信号更新 UI (单张结果通过 log 显示)
        elif os.path.isdir(input_path):
            extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            files = sorted(
                f for f in os.listdir(input_path)
                if os.path.splitext(f)[1].lower() in extensions
            )
            if not files:
                print("文件夹中未找到图片文件")
                return

            correct = 0
            total = len(files)

            for i, filename in enumerate(files):
                filepath = os.path.join(input_path, filename)
                result = predict_single(model, tokenizer, config, filepath, device)
                gt_label = os.path.splitext(filename)[0].split("_")[0].lower()
                match = result == gt_label
                if match:
                    correct += 1

                symbol = "O" if match else "X"
                print(f"  {symbol}  {filename}  ->  预测: {result}, 真实: {gt_label}")

            accuracy = correct / total if total > 0 else 0
            print(f"\n准确率: {correct}/{total} = {accuracy:.4f}")

    def _append_log(self, text):
        self.log_text.appendPlainText(text)

        # 解析日志行, 填充表格
        text = text.strip()
        if text.startswith(("O ", "X ")):
            # 格式: "O  filename  ->  预测: xxx, 真实: yyy"
            parts = text.split("->")
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()

                symbol = left[0]
                filename = left[2:].strip()

                pred = ""
                gt = ""
                for seg in right.split(","):
                    seg = seg.strip()
                    if seg.startswith("预测:"):
                        pred = seg.replace("预测:", "").strip()
                    elif seg.startswith("真实:"):
                        gt = seg.replace("真实:", "").strip()

                row = self.result_table.rowCount()
                self.result_table.insertRow(row)
                self.result_table.setItem(row, 0, QTableWidgetItem(filename))
                self.result_table.setItem(row, 1, QTableWidgetItem(pred))
                self.result_table.setItem(row, 2, QTableWidgetItem(gt))
                self.result_table.setItem(row, 3, QTableWidgetItem(symbol))

        elif text.startswith("识别结果:"):
            self.result_label.setText(text)

        elif text.startswith("准确率:"):
            self.accuracy_label.setText(text)

    def _on_finished(self, success, msg):
        self.predict_btn.setEnabled(True)
        self.progress_bar.setValue(100 if success else 0)
        if success:
            InfoBar.success("完成", "识别完成", parent=self,
                            position=InfoBarPosition.TOP, duration=3000)
        else:
            InfoBar.error("失败", msg, parent=self,
                          position=InfoBarPosition.TOP, duration=5000)
