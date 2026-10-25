"""本地文件读写抽象。"""

import hashlib
import os
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


def write_bytes_atomic(path: str | Path, data: bytes) -> None:
    """tmp+rename 原子写（基元）：读侧要么旧版要么新版，永无半份。

    tmp 是目标同目录的兄弟文件（同文件系统 rename 才原子）、点前缀隐藏命名
    （不进 glob/目录列举）；写 tmp 失败时目标分毫未动——丢弃 tmp 即回滚。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)


def write_text_atomic(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """文本门面：先编码后落盘——编码失败发生在任何文件效果之前（不留空 tmp 垃圾）。"""
    write_bytes_atomic(path, text.encode(encoding))


def sha256_of(path: str | Path, chunk: int = 8 * 1024 * 1024) -> str:
    """分块流式 sha256（file_hash 工具与指纹 tags 共用）。"""
    hasher = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            hasher.update(block)
    return hasher.hexdigest()


def sha256_text(text: str) -> str:
    """文本内容 sha256（yaml 明文输入的指纹面——无文件路径可哈希时的血缘真源）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_tags(kind: str, ref: str) -> dict[str, str]:
    """文件指纹 tags：``kind`` = data / modeling_yaml（MCP 边界到达的两种文件）。

    - ``py:target`` 引用哈希 ``:`` 前的文件（代码版本即数据版本）
    - 本地文件不存在（http 等远程引用）静默跳过——指纹只对可核验的本地工件负责
    """
    path = Path(ref.split(":", 1)[0])
    if not path.is_file():
        return {}
    return {f"mflowy.{kind}_sha256": sha256_of(path), f"mflowy.{kind}_file": ref}
