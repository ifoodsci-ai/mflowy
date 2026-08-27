"""info 工具行为测试 — file_hash / list_modules / get_module_info 返回 JSON 消息。"""

import hashlib

import pytest
from mflowy.mcp.tools import _CHUNK_SIZE, file_hash, get_module_info, list_modules


def test_file_hash_sha256_default(tmp_path):
    f = tmp_path / "data.csv"
    f.write_bytes(b"mflowy")
    assert file_hash(str(f)) == {
        "path": str(f),
        "algorithm": "sha256",
        "hash": hashlib.sha256(b"mflowy").hexdigest(),
        "size_bytes": 6,
    }


def test_file_hash_algorithm_variants_and_case_normalization(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"abc")
    md5 = file_hash(str(f), algorithm="md5")
    assert md5["algorithm"] == "md5"
    assert md5["hash"] == hashlib.md5(b"abc").hexdigest()
    # 枚举名/值双形式与大小写归一（SHA1 → sha1）
    assert file_hash(str(f), algorithm="SHA1")["hash"] == hashlib.sha1(b"abc").hexdigest()


def test_file_hash_streams_chunks(tmp_path):
    """超过 _CHUNK_SIZE 的文件走多块路径，size/哈希仍准确"""
    f = tmp_path / "big.bin"
    payload = b"\0" * (_CHUNK_SIZE + 1)
    f.write_bytes(payload)
    result = file_hash(str(f))
    assert result["size_bytes"] == len(payload)
    assert result["hash"] == hashlib.sha256(payload).hexdigest()


def test_file_hash_missing_file_returns_error():
    assert file_hash("/no/such/file.csv").startswith("Error: File not found")


def test_file_hash_directory_returns_error(tmp_path):
    assert file_hash(str(tmp_path)).startswith("Error: Path is not a file")


def test_file_hash_unsupported_algorithm(tmp_path):
    """Literal 约束 schema；直调传入非法值时返回可读错误而非抛异常"""
    f = tmp_path / "f"
    f.write_bytes(b"1")
    msg = file_hash(str(f), algorithm="crc32")  # type: ignore[arg-type]
    assert msg.startswith("Error: Unsupported algorithm")
    assert "sha1" in msg and "sha256" in msg


def test_list_modules_groups():
    data = list_modules()
    steps = {item["step"] for item in data}
    assert "model" in steps and "load" in steps
    model = next(i for i in data if i["step"] == "model")
    assert "XGB" in model["modules"]


def test_list_modules_filter_by_step():
    data = list_modules(step="model")
    assert [i["step"] for i in data] == ["model"]


def test_list_modules_invalid_step_lists_valid_options():
    with pytest.raises(ModuleNotFoundError, match="nope"):
        list_modules(step="nope")


def test_get_module_info_happy_path():
    info = get_module_info("model", "XGB")
    assert info["name"] == "model.XGB"
    assert "XGBoost" in info["description"]  # doc 首行
    names = [p["name"] for p in info["parameters"]]
    assert "n_estimators" in names and "scoring" in names
    n_trials = next(p for p in info["parameters"] if p["name"] == "n_trials")
    assert n_trials["required"] is False
    assert n_trials["default"] == 100
    assert len(n_trials["description"]) > 0


def test_get_module_info_required_param_default_is_none():
    """必填参数 default 恒为 None——不外泄 inspect._empty 哨兵（结构化契约）"""
    info = get_module_info("load", "csv")
    source = next(p for p in info["parameters"] if p["name"] == "source")
    assert source["required"] is True
    assert source["default"] is None


def test_get_module_info_unknown_module():
    with pytest.raises(ModuleNotFoundError):
        get_module_info("model", "NoSuchModule")
