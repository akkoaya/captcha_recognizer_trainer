"""
字符编解码器

将验证码中的字符与数字索引相互转换。
例如: "a3Kp" → [13, 6, 23, 28] → "a3Kp"

词表结构:
  索引 0: [BLANK]  — CTC 空白符 / 填充符
  索引 1: [SOS]    — 序列开始标记 (Attention 模式使用)
  索引 2: [EOS]    — 序列结束标记 (Attention 模式使用)
  索引 3+: 实际字符 (0-9, a-z, ...)
"""


class Tokenizer:
    """字符编解码器：负责字符与数字索引之间的相互转换"""

    def __init__(self, charset: str):
        """
        Args:
            charset: 字符集字符串，如 "0123456789abcdefghijklmnopqrstuvwxyz"
        """
        # 去重并排序，保证顺序稳定
        self.charset = sorted(set(charset))

        # 特殊标记
        self.special_tokens = ["[BLANK]", "[SOS]", "[EOS]"]

        # 完整词表 = 特殊标记 + 实际字符
        self.vocab = self.special_tokens + self.charset

        # 双向映射表
        self.char_to_index = {char: idx for idx, char in enumerate(self.vocab)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        # 特殊标记的索引
        self.blank_index = 0   # CTC 空白符
        self.sos_index = 1     # 序列开始
        self.eos_index = 2     # 序列结束
        self.pad_index = 0     # 填充 (与 blank 共享索引 0)

    @property
    def vocab_size(self) -> int:
        """词表大小 (包含特殊标记)"""
        return len(self.vocab)

    def encode(self, text: str) -> list:
        """
        将字符串编码为索引列表

        Args:
            text: 输入字符串，如 "a3kp"

        Returns:
            索引列表，如 [13, 6, 23, 28]
        """
        return [self.char_to_index[c] for c in text if c in self.char_to_index]

    def decode_ctc(self, indices: list) -> str:
        """
        CTC 解码：合并连续重复字符，删除空白符

        CTC 输出示例: [0, 13, 13, 0, 6, 6, 6, 0, 23, 0]
        合并重复:      [0, 13, 0, 6, 0, 23, 0]
        删除空白(0):   [13, 6, 23]
        转为字符:      "a3k"

        Args:
            indices: 模型输出的索引序列

        Returns:
            解码后的字符串
        """
        result = []
        prev_idx = -1
        for idx in indices:
            # 跳过空白符和填充
            if idx == self.blank_index:
                prev_idx = idx
                continue
            # 合并连续重复的字符
            if idx != prev_idx:
                char = self.index_to_char.get(idx, "")
                if char not in self.special_tokens:
                    result.append(char)
            prev_idx = idx
        return "".join(result)

    def decode_attention(self, indices: list) -> str:
        """
        Attention 解码：遇到 EOS 就停止

        Args:
            indices: 模型输出的索引序列

        Returns:
            解码后的字符串
        """
        result = []
        for idx in indices:
            if idx == self.eos_index:
                break
            char = self.index_to_char.get(idx, "")
            if char not in self.special_tokens:
                result.append(char)
        return "".join(result)

    def decode(self, indices: list, mode: str = "ctc") -> str:
        """
        统一解码接口

        Args:
            indices: 索引序列
            mode: "ctc" 或 "attention"

        Returns:
            解码后的字符串
        """
        if mode == "ctc":
            return self.decode_ctc(indices)
        else:
            return self.decode_attention(indices)
