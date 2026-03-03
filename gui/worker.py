"""
后台工作线程

将耗时任务 (训练、生成、导出、推理) 放到子线程执行,
通过信号将 stdout/stderr 输出实时转发到 GUI 日志区域。
"""

import sys
import io
import traceback
from PyQt5.QtCore import QThread, pyqtSignal


class LogStream(io.TextIOBase):
    """将写入内容转发到 Qt 信号的文本流"""

    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def write(self, text):
        if text and text.strip():
            self._signal.emit(text)
        return len(text) if text else 0

    def flush(self):
        pass


class WorkerThread(QThread):
    """
    通用后台工作线程

    Signals:
        log_signal: 日志输出 (str)
        finished_signal: 任务结束 (bool 成功, str 消息)
        progress_signal: 进度 (int 0-100)
    """

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    progress_signal = pyqtSignal(int)

    def __init__(self, target, args=(), kwargs=None):
        """
        Args:
            target: 要执行的函数
            args:   位置参数
            kwargs: 关键字参数
        """
        super().__init__()
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self._stop_flag = False

    def request_stop(self):
        """请求停止任务"""
        self._stop_flag = True

    @property
    def should_stop(self):
        return self._stop_flag

    def run(self):
        log_stream = LogStream(self.log_signal)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = log_stream
        sys.stderr = log_stream
        try:
            self._target(*self._args, **self._kwargs)
            self.finished_signal.emit(True, "任务完成")
        except Exception:
            err = traceback.format_exc()
            self.log_signal.emit(err)
            self.finished_signal.emit(False, "任务失败")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
