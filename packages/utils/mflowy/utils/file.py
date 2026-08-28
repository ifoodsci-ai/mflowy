"""本地文件读取抽象。"""

import hashlib
from pathlib import Path


def read_bytes(path: str) -> bytes:
    """读取文件全部内容为 bytes。"""
    return Path(path).read_bytes()


def read_text(path: str, encoding: str = "utf-8") -> str:
    """读取文件全部内容为文本。"""
    return Path(path).read_text(encoding=encoding)


def exists(path: str) -> bool:
    """检查文件是否存在。"""
    return Path(path).exists()


def sha256_of(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    """分块流式 sha256（file_hash 工具与指纹 tags 共用）。"""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            hasher.update(block)
    return hasher.hexdigest()


def fingerprint_tags(kind: str, ref: str) -> dict[str, str]:
    """文件指纹 tags：``kind`` = data / modeling_yaml（MCP 边界到达的两种文件）。

    - ``py:target`` 引用哈希 ``:`` 前的文件（代码版本即数据版本）
    - 本地文件不存在（http 等远程引用）静默跳过——指纹只对可核验的本地工件负责
    """
    path = Path(ref.split(":", 1)[0])
    if not path.is_file():
        return {}
    return {f"mflowy.{kind}_sha256": sha256_of(path), f"mflowy.{kind}_file": ref}
