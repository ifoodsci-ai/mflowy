"""指纹件（mflowy.utils.file）：sha256_of 流式哈希 + fingerprint_tags（data / modeling_yaml 两种文件）"""

import hashlib

from mflowy.utils.file import fingerprint_tags, sha256_of


def test_sha256_of_streams(tmp_path):
    f = tmp_path / "data.csv"
    payload = b"x" * (8 * 1024 * 1024 + 7)  # 跨块边界
    f.write_bytes(payload)
    assert sha256_of(f) == hashlib.sha256(payload).hexdigest()


def test_fingerprint_tags_local_file(tmp_path):
    f = tmp_path / "diabetes.csv"
    f.write_bytes(b"age,target\n1,2\n")
    tags = fingerprint_tags("data", str(f))
    assert tags["mflowy.data_sha256"] == hashlib.sha256(b"age,target\n1,2\n").hexdigest()
    assert tags["mflowy.data_file"] == str(f)


def test_fingerprint_tags_py_target_hashes_code_file(tmp_path):
    """py:target 引用哈希 : 前的文件——代码版本即数据版本"""
    code = tmp_path / "loader.py"
    code.write_text("def load(): pass")
    tags = fingerprint_tags("data", f"{code}:load")
    assert tags["mflowy.data_sha256"] == hashlib.sha256(b"def load(): pass").hexdigest()
    assert tags["mflowy.data_file"] == f"{code}:load"


def test_fingerprint_tags_missing_file_skipped():
    """远程引用 / 不存在的文件静默跳过——指纹只对可核验的本地工件负责"""
    assert fingerprint_tags("data", "http://example.com/x.csv") == {}
    assert fingerprint_tags("modeling_yaml", "/no/such.yaml") == {}


def test_fingerprint_tags_kind_naming(tmp_path):
    f = tmp_path / "steps.yaml"
    f.write_text("steps: []")
    tags = fingerprint_tags("modeling_yaml", str(f))
    assert "mflowy.modeling_yaml_sha256" in tags and "mflowy.modeling_yaml_file" in tags
