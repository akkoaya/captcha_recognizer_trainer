"""
数据生成页面

提供验证码数据集的生成功能, 对应 tools/generate_captcha.py。
可配置字符集、长度范围、数量、图片尺寸等参数。
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
)
from qfluentwidgets import (
    ScrollArea, TitleLabel, SubtitleLabel, BodyLabel,
    LineEdit, SpinBox, PrimaryPushButton, PushButton,
    CardWidget, InfoBar, InfoBarPosition, ProgressBar,
    FluentIcon as FIF,
)

from gui.worker import WorkerThread


class GeneratePage(ScrollArea):
    """数据生成页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("generate_page")
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
        layout.addWidget(TitleLabel("数据生成"))
        layout.addWidget(BodyLabel("使用 captcha 库生成训练和验证数据集, 图片文件名格式: 标签_序号.png"))

        # ── 字符集配置 ──
        charset_card = CardWidget(self.scroll_widget)
        charset_layout = QVBoxLayout(charset_card)
        charset_layout.setContentsMargins(20, 16, 20, 16)
        charset_layout.addWidget(SubtitleLabel("字符集"))

        self.charset_edit = LineEdit()
        self.charset_edit.setText("0123456789abcdefghijklmnopqrstuvwxyz")
        self.charset_edit.setPlaceholderText("输入模型能识别的所有字符")
        charset_layout.addWidget(self.charset_edit)
        layout.addWidget(charset_card)

        # ── 长度 & 数量 ──
        params_card = CardWidget(self.scroll_widget)
        params_layout = QVBoxLayout(params_card)
        params_layout.setContentsMargins(20, 16, 20, 16)
        params_layout.addWidget(SubtitleLabel("生成参数"))

        row1 = QHBoxLayout()

        min_len_layout = QVBoxLayout()
        min_len_layout.addWidget(BodyLabel("最小长度"))
        self.min_length_spin = SpinBox()
        self.min_length_spin.setRange(1, 20)
        self.min_length_spin.setValue(4)
        min_len_layout.addWidget(self.min_length_spin)
        row1.addLayout(min_len_layout)

        max_len_layout = QVBoxLayout()
        max_len_layout.addWidget(BodyLabel("最大长度"))
        self.max_length_spin = SpinBox()
        self.max_length_spin.setRange(1, 20)
        self.max_length_spin.setValue(6)
        max_len_layout.addWidget(self.max_length_spin)
        row1.addLayout(max_len_layout)

        width_layout = QVBoxLayout()
        width_layout.addWidget(BodyLabel("图片宽度"))
        self.width_spin = SpinBox()
        self.width_spin.setRange(32, 512)
        self.width_spin.setValue(160)
        width_layout.addWidget(self.width_spin)
        row1.addLayout(width_layout)

        height_layout = QVBoxLayout()
        height_layout.addWidget(BodyLabel("图片高度"))
        self.height_spin = SpinBox()
        self.height_spin.setRange(16, 256)
        self.height_spin.setValue(64)
        height_layout.addWidget(self.height_spin)
        row1.addLayout(height_layout)

        params_layout.addLayout(row1)

        row2 = QHBoxLayout()

        train_num_layout = QVBoxLayout()
        train_num_layout.addWidget(BodyLabel("训练集数量"))
        self.train_num_spin = SpinBox()
        self.train_num_spin.setRange(100, 999999)
        self.train_num_spin.setValue(20000)
        train_num_layout.addWidget(self.train_num_spin)
        row2.addLayout(train_num_layout)

        val_num_layout = QVBoxLayout()
        val_num_layout.addWidget(BodyLabel("验证集数量"))
        self.val_num_spin = SpinBox()
        self.val_num_spin.setRange(100, 999999)
        self.val_num_spin.setValue(2000)
        val_num_layout.addWidget(self.val_num_spin)
        row2.addLayout(val_num_layout)

        params_layout.addLayout(row2)

        # 输出路径
        row3 = QHBoxLayout()

        train_dir_layout = QVBoxLayout()
        train_dir_layout.addWidget(BodyLabel("训练集目录"))
        self.train_dir_edit = LineEdit()
        self.train_dir_edit.setText("data/train")
        train_dir_layout.addWidget(self.train_dir_edit)
        row3.addLayout(train_dir_layout)

        val_dir_layout = QVBoxLayout()
        val_dir_layout.addWidget(BodyLabel("验证集目录"))
        self.val_dir_edit = LineEdit()
        self.val_dir_edit.setText("data/val")
        val_dir_layout.addWidget(self.val_dir_edit)
        row3.addLayout(val_dir_layout)

        params_layout.addLayout(row3)
        layout.addWidget(params_card)

        # ── 进度 & 操作 ──
        action_card = CardWidget(self.scroll_widget)
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(20, 16, 20, 16)

        self.progress_bar = ProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.generate_btn = PrimaryPushButton(FIF.PLAY, "开始生成")
        self.generate_btn.clicked.connect(self._on_generate)
        btn_layout.addWidget(self.generate_btn)
        action_layout.addLayout(btn_layout)

        layout.addWidget(action_card)

        # ── 日志 ──
        log_card = CardWidget(self.scroll_widget)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(20, 16, 20, 16)
        log_layout.addWidget(SubtitleLabel("日志"))

        from PyQt6.QtWidgets import QPlainTextEdit
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(180)
        self.log_text.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #cccccc;"
            " font-family: Consolas, 'Courier New', monospace; font-size: 13px;"
            " border-radius: 8px; padding: 8px; }"
        )
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_card)

        layout.addStretch()

    def _on_generate(self):
        if self.worker and self.worker.isRunning():
            InfoBar.warning("提示", "生成任务正在运行中", parent=self,
                            position=InfoBarPosition.TOP, duration=2000)
            return

        # 参数校验
        min_len = self.min_length_spin.value()
        max_len = self.max_length_spin.value()
        if min_len > max_len:
            InfoBar.error("参数错误", "最小长度不能大于最大长度", parent=self,
                          position=InfoBarPosition.TOP, duration=3000)
            return

        charset = self.charset_edit.text().strip()
        if not charset:
            InfoBar.error("参数错误", "字符集不能为空", parent=self,
                          position=InfoBarPosition.TOP, duration=3000)
            return

        self.log_text.clear()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.generate_btn.setEnabled(False)

        kwargs = dict(
            charset=charset,
            min_length=min_len,
            max_length=max_len,
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            train_dir=self.train_dir_edit.text().strip(),
            val_dir=self.val_dir_edit.text().strip(),
            num_train=self.train_num_spin.value(),
            num_val=self.val_num_spin.value(),
        )

        self.worker = WorkerThread(target=self._run_generate, kwargs=kwargs)
        self.worker.log_signal.connect(self._append_log)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    @staticmethod
    def _run_generate(charset, min_length, max_length, width, height,
                      train_dir, val_dir, num_train, num_val):
        from tools.generate_captcha import generate_dataset

        generate_dataset(
            output_dir=train_dir,
            num_samples=num_train,
            charset=charset,
            min_length=min_length,
            max_length=max_length,
            width=width,
            height=height,
        )
        generate_dataset(
            output_dir=val_dir,
            num_samples=num_val,
            charset=charset,
            min_length=min_length,
            max_length=max_length,
            width=width,
            height=height,
        )

    def _append_log(self, text):
        self.log_text.appendPlainText(text)

    def _on_finished(self, success, msg):
        self.generate_btn.setEnabled(True)
        self.progress_bar.setValue(100 if success else 0)
        if success:
            InfoBar.success("完成", "数据集生成完毕", parent=self,
                            position=InfoBarPosition.TOP, duration=3000)
        else:
            InfoBar.error("失败", msg, parent=self,
                          position=InfoBarPosition.TOP, duration=5000)
