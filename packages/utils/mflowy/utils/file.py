"""本地文件读取抽象。"""

from pathlib import Path


def read_bytes(path: str | Path) -> bytes:
    """读取文件全部内容为 bytes。"""
    return Path(path).read_bytes()


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    """读取文件全部内容为文本。"""
    return Path(path).read_text(encoding=encoding)


def exists(path: str | Path) -> bool:
    """检查文件是否存在。"""
    return Path(path).exists()
