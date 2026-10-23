"""帧内容指纹 CAS 契约：df → 规范 parquet 字节流（``index=False``）→ sha256——指纹即内容地址。

与文件级 ``sha256_of`` 的关系由 ``test_cas_address_equals_artifact_sha256`` 钉死：
对同参数序列化（``to_parquet(index=False)``）落盘的工件，指纹 = 工件 sha256——
两个粒度在"规范字节表示"上合一（CAS 闭合），行序进指纹、行标签不进。
"""

import re

import pandas as pd
from mflowy.data.fingerprint import fingerprint

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_cas_address_deterministic_hex():
    """同内容（两次独立构造）→ 同地址；地址形如 sha256 十六进制（64 位小写）。"""
    a = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    b = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    assert fingerprint(a) == fingerprint(b)
    assert _HEX64.match(fingerprint(a))


def test_value_drift_changes_address():
    """值内容变 → 地址变（防错配守卫的基本盘）。"""
    a = pd.DataFrame({"a": [1.0, 2.0]})
    b = pd.DataFrame({"a": [1.0, 9.0]})
    assert fingerprint(a) != fingerprint(b)


def test_row_order_sensitive():
    """行序敏感（CAS 语义翻转：行序是字节内容的一部分）——重排同内容集得到不同地址。"""
    a = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert fingerprint(a) != fingerprint(a.iloc[::-1])


def test_index_labels_invisible():
    """行标签不进指纹：``index=False`` 序列化——同值不同标签（重放侧索引差）不误报。"""
    a = pd.DataFrame({"a": [1.0, 2.0]})
    b = a.copy()
    b.index = [10, 20]
    assert fingerprint(a) == fingerprint(b)


def test_zero_column_frame_counts_rows():
    """零列帧无值可寻址——内容退化为行数（sha256 仍为 64 位 hex，格式统一）。"""
    assert fingerprint(pd.DataFrame(index=range(3))) != fingerprint(pd.DataFrame(index=range(5)))
    assert fingerprint(pd.DataFrame(index=range(3))) == fingerprint(pd.DataFrame(index=range(3)))
    assert _HEX64.match(fingerprint(pd.DataFrame(index=range(3))))


def test_cas_address_equals_artifact_sha256(tmp_path):
    """CAS 闭合：指纹 = 规范序列化（``to_parquet(index=False)``）落盘工件的 sha256——
    独立真源走盘上文件（``mflowy.utils.file.sha256_of``），不经内存 buffer。"""
    from mflowy.utils.file import sha256_of

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    p = tmp_path / "v0.parquet"
    df.to_parquet(p, index=False)
    assert fingerprint(df) == sha256_of(p)
