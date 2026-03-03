"""
模型训练页面

提供完整的训练参数配置和实时训练监控, 对应 train.py。
支持从 config.yaml 加载默认配置, 也可在 GUI 中修改后启动训练。
"""

import os
import yaml
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit,
)
from qfluentwidgets import (
    ScrollArea, TitleLabel, SubtitleLabel, BodyLabel,
    LineEdit, SpinBox, DoubleSpinBox, ComboBox,
    PrimaryPushButton, PushButton, CardWidget,
    InfoBar, InfoBarPosition, ProgressBar,
    CheckBox, FluentIcon as FIF,
)

from gui.worker import WorkerThread


class TrainPage(ScrollArea):
    """模型训练页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("train_page")
        self.worker = None

        self._init_ui()
        self._load_config()

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
        layout.addWidget(TitleLabel("模型训练"))
        layout.addWidget(BodyLabel("配置训练参数并启动训练, 实时查看训练日志和指标"))

        # ── 模型配置 ──
        model_card = CardWidget(self.scroll_widget)
        model_layout = QVBoxLayout(model_card)
        model_layout.setContentsMargins(20, 16, 20, 16)
        model_layout.addWidget(SubtitleLabel("模型配置"))

        row = QHBoxLayout()

        backbone_layout = QVBoxLayout()
        backbone_layout.addWidget(BodyLabel("骨干网络"))
        self.backbone_combo = ComboBox()
        self.backbone_combo.addItems(["resnet"])
        backbone_layout.addWidget(self.backbone_combo)
        row.addLayout(backbone_layout)

        head_layout = QVBoxLayout()
        head_layout.addWidget(BodyLabel("识别头"))
        self.head_combo = ComboBox()
        self.head_combo.addItems(["ctc", "attention"])
        head_layout.addWidget(self.head_combo)
        row.addLayout(head_layout)

        dim_layout = QVBoxLayout()
        dim_layout.addWidget(BodyLabel("特征维度"))
        self.feature_dim_spin = SpinBox()
        self.feature_dim_spin.setRange(64, 1024)
        self.feature_dim_spin.setValue(256)
        self.feature_dim_spin.setSingleStep(64)
        dim_layout.addWidget(self.feature_dim_spin)
        row.addLayout(dim_layout)

        model_layout.addLayout(row)
        layout.addWidget(model_card)

        # ── 数据配置 ──
        data_card = CardWidget(self.scroll_widget)
        data_layout = QVBoxLayout(data_card)
        data_layout.setContentsMargins(20, 16, 20, 16)
        data_layout.addWidget(SubtitleLabel("数据配置"))

        row_charset = QHBoxLayout()
        charset_col = QVBoxLayout()
        charset_col.addWidget(BodyLabel("字符集"))
        self.charset_edit = LineEdit()
        self.charset_edit.setText("0123456789abcdefghijklmnopqrstuvwxyz")
        charset_col.addWidget(self.charset_edit)
        row_charset.addLayout(charset_col)
        data_layout.addLayout(row_charset)

        row_img = QHBoxLayout()

        for label_text, attr_name, default_val, min_val, max_val in [
            ("图片高度", "img_height_spin", 64, 16, 512),
            ("图片宽度", "img_width_spin", 160, 32, 1024),
            ("通道数", "img_channel_spin", 1, 1, 3),
            ("最小标签长度", "min_label_spin", 4, 1, 50),
            ("最大标签长度", "max_label_spin", 10, 1, 50),
        ]:
            col = QVBoxLayout()
            col.addWidget(BodyLabel(label_text))
            spin = SpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default_val)
            setattr(self, attr_name, spin)
            col.addWidget(spin)
            row_img.addLayout(col)

        data_layout.addLayout(row_img)

        row_dirs = QHBoxLayout()
        train_dir_col = QVBoxLayout()
        train_dir_col.addWidget(BodyLabel("训练集目录"))
        self.train_dir_edit = LineEdit()
        self.train_dir_edit.setText("data/train")
        train_dir_col.addWidget(self.train_dir_edit)
        row_dirs.addLayout(train_dir_col)

        val_dir_col = QVBoxLayout()
        val_dir_col.addWidget(BodyLabel("验证集目录"))
        self.val_dir_edit = LineEdit()
        self.val_dir_edit.setText("data/val")
        val_dir_col.addWidget(self.val_dir_edit)
        row_dirs.addLayout(val_dir_col)

        data_layout.addLayout(row_dirs)
        layout.addWidget(data_card)

        # ── 训练配置 ──
        train_card = CardWidget(self.scroll_widget)
        train_layout = QVBoxLayout(train_card)
        train_layout.setContentsMargins(20, 16, 20, 16)
        train_layout.addWidget(SubtitleLabel("训练配置"))

        row_train1 = QHBoxLayout()
        for label_text, attr_name, default_val, min_val, max_val in [
            ("Epochs", "epochs_spin", 100, 1, 9999),
            ("Batch Size", "batch_spin", 128, 1, 2048),
            ("Patience", "patience_spin", 15, 1, 999),
            ("Workers", "workers_spin", 4, 0, 32),
        ]:
            col = QVBoxLayout()
            col.addWidget(BodyLabel(label_text))
            spin = SpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default_val)
            setattr(self, attr_name, spin)
            col.addWidget(spin)
            row_train1.addLayout(col)
        train_layout.addLayout(row_train1)

        row_train2 = QHBoxLayout()

        lr_col = QVBoxLayout()
        lr_col.addWidget(BodyLabel("学习率"))
        self.lr_spin = DoubleSpinBox()
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.0001)
        lr_col.addWidget(self.lr_spin)
        row_train2.addLayout(lr_col)

        wd_col = QVBoxLayout()
        wd_col.addWidget(BodyLabel("权重衰减"))
        self.wd_spin = DoubleSpinBox()
        self.wd_spin.setRange(0.0, 1.0)
        self.wd_spin.setValue(0.0001)
        self.wd_spin.setDecimals(6)
        self.wd_spin.setSingleStep(0.0001)
        wd_col.addWidget(self.wd_spin)
        row_train2.addLayout(wd_col)

        gc_col = QVBoxLayout()
        gc_col.addWidget(BodyLabel("梯度裁剪"))
        self.grad_clip_spin = DoubleSpinBox()
        self.grad_clip_spin.setRange(0.1, 100.0)
        self.grad_clip_spin.setValue(5.0)
        self.grad_clip_spin.setDecimals(1)
        gc_col.addWidget(self.grad_clip_spin)
        row_train2.addLayout(gc_col)

        train_layout.addLayout(row_train2)

        row_train3 = QHBoxLayout()
        self.amp_check = CheckBox("混合精度训练 (AMP)")
        self.amp_check.setChecked(True)
        row_train3.addWidget(self.amp_check)
        row_train3.addStretch()

        save_col = QVBoxLayout()
        save_col.addWidget(BodyLabel("模型保存路径"))
        self.save_dir_edit = LineEdit()
        self.save_dir_edit.setText("checkpoints")
        save_col.addWidget(self.save_dir_edit)
        row_train3.addLayout(save_col)

        train_layout.addLayout(row_train3)
        layout.addWidget(train_card)

        # ── 数据增强 ──
        aug_card = CardWidget(self.scroll_widget)
        aug_layout = QVBoxLayout(aug_card)
        aug_layout.setContentsMargins(20, 16, 20, 16)
        aug_layout.addWidget(SubtitleLabel("数据增强"))

        row_aug1 = QHBoxLayout()

        rot_col = QVBoxLayout()
        rot_col.addWidget(BodyLabel("旋转角度"))
        self.rotate_spin = SpinBox()
        self.rotate_spin.setRange(0, 45)
        self.rotate_spin.setValue(10)
        rot_col.addWidget(self.rotate_spin)
        row_aug1.addLayout(rot_col)

        blur_col = QVBoxLayout()
        blur_col.addWidget(BodyLabel("模糊核大小"))
        self.blur_spin = SpinBox()
        self.blur_spin.setRange(1, 15)
        self.blur_spin.setValue(3)
        blur_col.addWidget(self.blur_spin)
        row_aug1.addLayout(blur_col)

        aug_layout.addLayout(row_aug1)

        row_aug2 = QHBoxLayout()
        self.noise_check = CheckBox("高斯噪声")
        self.noise_check.setChecked(True)
        row_aug2.addWidget(self.noise_check)

        self.perspective_check = CheckBox("透视变形")
        self.perspective_check.setChecked(True)
        row_aug2.addWidget(self.perspective_check)

        self.brightness_check = CheckBox("随机亮度对比度")
        self.brightness_check.setChecked(True)
        row_aug2.addWidget(self.brightness_check)

        row_aug2.addStretch()
        aug_layout.addLayout(row_aug2)
        layout.addWidget(aug_card)

        # ── 操作按钮 ──
        action_card = CardWidget(self.scroll_widget)
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(20, 16, 20, 16)

        self.progress_bar = ProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.load_cfg_btn = PushButton(FIF.SYNC, "加载配置")
        self.load_cfg_btn.clicked.connect(self._load_config)
        btn_layout.addWidget(self.load_cfg_btn)

        self.stop_btn = PushButton(FIF.CLOSE, "停止训练")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_layout.addWidget(self.stop_btn)

        self.train_btn = PrimaryPushButton(FIF.PLAY, "开始训练")
        self.train_btn.clicked.connect(self._on_train)
        btn_layout.addWidget(self.train_btn)

        action_layout.addLayout(btn_layout)
        layout.addWidget(action_card)

        # ── 日志 ──
        log_card = CardWidget(self.scroll_widget)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(20, 16, 20, 16)
        log_layout.addWidget(SubtitleLabel("训练日志"))

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(260)
        self.log_text.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #cccccc;"
            " font-family: Consolas, 'Courier New', monospace; font-size: 13px;"
            " border-radius: 8px; padding: 8px; }"
        )
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_card)

        layout.addStretch()

    def _load_config(self):
        """从 config.yaml 加载配置到各控件"""
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception:
            return

        m = cfg.get("model", {})
        d = cfg.get("data", {})
        t = cfg.get("train", {})
        a = cfg.get("augment", {})

        self.backbone_combo.setCurrentText(m.get("backbone", "resnet"))
        self.head_combo.setCurrentText(m.get("head", "ctc"))
        self.feature_dim_spin.setValue(m.get("feature_dim", 256))

        self.charset_edit.setText(d.get("charset", ""))
        self.img_height_spin.setValue(d.get("image_height", 64))
        self.img_width_spin.setValue(d.get("image_width", 160))
        self.img_channel_spin.setValue(d.get("image_channel", 1))
        self.min_label_spin.setValue(d.get("min_label_length", 4))
        self.max_label_spin.setValue(d.get("max_label_length", 10))
        self.train_dir_edit.setText(d.get("train_dir", "data/train"))
        self.val_dir_edit.setText(d.get("val_dir", "data/val"))

        self.epochs_spin.setValue(t.get("epochs", 100))
        self.batch_spin.setValue(t.get("batch_size", 128))
        self.lr_spin.setValue(t.get("lr", 0.001))
        self.wd_spin.setValue(t.get("weight_decay", 0.0001))
        self.patience_spin.setValue(t.get("patience", 15))
        self.workers_spin.setValue(t.get("num_workers", 4))
        self.amp_check.setChecked(t.get("use_amp", True))
        self.grad_clip_spin.setValue(t.get("grad_clip", 5.0))
        self.save_dir_edit.setText(t.get("save_dir", "checkpoints"))

        self.rotate_spin.setValue(a.get("rotate_limit", 10))
        self.blur_spin.setValue(a.get("blur_limit", 3))
        self.noise_check.setChecked(a.get("noise", True))
        self.perspective_check.setChecked(a.get("perspective", True))
        self.brightness_check.setChecked(a.get("brightness_contrast", True))

    def _build_config(self) -> dict:
        """从 GUI 控件构建 config 字典"""
        return {
            "model": {
                "backbone": self.backbone_combo.currentText(),
                "head": self.head_combo.currentText(),
                "feature_dim": self.feature_dim_spin.value(),
            },
            "data": {
                "charset": self.charset_edit.text().strip(),
                "image_height": self.img_height_spin.value(),
                "image_width": self.img_width_spin.value(),
                "image_channel": self.img_channel_spin.value(),
                "min_label_length": self.min_label_spin.value(),
                "max_label_length": self.max_label_spin.value(),
                "train_dir": self.train_dir_edit.text().strip(),
                "val_dir": self.val_dir_edit.text().strip(),
            },
            "train": {
                "epochs": self.epochs_spin.value(),
                "batch_size": self.batch_spin.value(),
                "lr": self.lr_spin.value(),
                "weight_decay": self.wd_spin.value(),
                "patience": self.patience_spin.value(),
                "num_workers": self.workers_spin.value(),
                "use_amp": self.amp_check.isChecked(),
                "grad_clip": self.grad_clip_spin.value(),
                "save_dir": self.save_dir_edit.text().strip(),
            },
            "augment": {
                "rotate_limit": self.rotate_spin.value(),
                "blur_limit": self.blur_spin.value(),
                "noise": self.noise_check.isChecked(),
                "perspective": self.perspective_check.isChecked(),
                "brightness_contrast": self.brightness_check.isChecked(),
            },
        }

    def _on_train(self):
        if self.worker and self.worker.isRunning():
            InfoBar.warning("提示", "训练任务正在运行中", parent=self,
                            position=InfoBarPosition.TOP, duration=2000)
            return

        config = self._build_config()

        # 校验
        if not config["data"]["charset"]:
            InfoBar.error("参数错误", "字符集不能为空", parent=self,
                          position=InfoBarPosition.TOP, duration=3000)
            return

        if not os.path.isdir(config["data"]["train_dir"]):
            InfoBar.error("路径错误", f"训练集目录不存在: {config['data']['train_dir']}",
                          parent=self, position=InfoBarPosition.TOP, duration=3000)
            return

        if not os.path.isdir(config["data"]["val_dir"]):
            InfoBar.error("路径错误", f"验证集目录不存在: {config['data']['val_dir']}",
                          parent=self, position=InfoBarPosition.TOP, duration=3000)
            return

        self.log_text.clear()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = WorkerThread(target=self._run_train, args=(config,))
        self.worker.log_signal.connect(self._append_log)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    @staticmethod
    def _run_train(config):
        """在子线程中执行训练"""
        import os
        import yaml
        import tempfile

        # 将 GUI 配置写入临时文件, 然后调用 train 函数
        tmp_path = os.path.join(tempfile.gettempdir(), "captcha_train_cfg.yaml")
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)

        from train import train
        train(tmp_path)

    def _on_stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(3000)
            self._append_log("\n训练已被用户中断。")
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)

    def _append_log(self, text):
        self.log_text.appendPlainText(text)
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_finished(self, success, msg):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)
        if success:
            InfoBar.success("完成", "训练完成!", parent=self,
                            position=InfoBarPosition.TOP, duration=5000)
        else:
            InfoBar.error("失败", msg, parent=self,
                          position=InfoBarPosition.TOP, duration=5000)
