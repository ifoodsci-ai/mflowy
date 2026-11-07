"""write_bytes_atomic / write_text_atomic 契约：tmp+rename 原子写的机器证明。

三个不变式：成功=新内容且无垃圾残留；写 tmp 失败=目标分毫未动且垃圾是点前缀
兄弟；编码失败=零文件效果（先编码后落盘）。
"""

from pathlib import Path

import pytest
from mflowy.utils.file import write_bytes_atomic, write_text_atomic


def test_success_writes_content_and_leaves_no_garbage(tmp_path):
    p = tmp_path / "m.json"
    write_bytes_atomic(p, b"v1")
    assert p.read_bytes() == b"v1"
    assert list(tmp_path.iterdir()) == [p]  # tmp 被 rename 吃掉


def test_overwrite_yields_new_whole_version(tmp_path):
    p = tmp_path / "m.json"
    write_bytes_atomic(p, b"v1")
    write_bytes_atomic(p, b"v2-longer")
    assert p.read_bytes() == b"v2-longer"  # 整体替换，无残留拼接


def test_torn_write_leaves_original_intact(tmp_path, monkeypatch):
    p = tmp_path / "m.bin"
    write_bytes_atomic(p, b"old")
    real = Path.write_bytes

    def torn(self, data):  # 模拟写一半 ENOSPC：tmp 落半份后抛
        real(self, data[:2])
        raise OSError("ENOSPC")

    monkeypatch.setattr(Path, "write_bytes", torn)
    with pytest.raises(OSError):
        write_bytes_atomic(p, b"newdata")
    assert p.read_bytes() == b"old"  # 目标分毫未动
    assert (tmp_path / ".m.bin.tmp").exists()  # 垃圾是点前缀兄弟，一望即知


def test_creates_parent_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "m.json"
    write_text_atomic(p, "x")
    assert p.read_text() == "x"


def test_text_roundtrip_unicode(tmp_path):
    p = tmp_path / "m.yaml"
    write_text_atomic(p, "阈值: ≥0.5")
    assert p.read_text(encoding="utf-8") == "阈值: ≥0.5"


def test_encode_failure_has_zero_file_effect(tmp_path):
    p = tmp_path / "m.json"
    with pytest.raises(UnicodeEncodeError):
        write_text_atomic(p, "\ud800")  # lone surrogate 无法 utf-8 编码
    assert list(tmp_path.iterdir()) == []  # 先编码后落盘：连 tmp 都没出生
