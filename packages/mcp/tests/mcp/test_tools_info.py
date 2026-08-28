"""info 工具行为测试 — file_hash / list_modules / get_module_info 返回 JSON 消息。"""

import hashlib

import pytest
from mflowy.mcp.tools import file_hash, get_module_info, list_modules


def test_file_hash_sha256_default(tmp_path):
    f = tmp_path / "data.csv"
    f.write_bytes(b"mflowy")
    assert file_hash(str(f)) == {
        "path": str(f),
        "sha256": hashlib.sha256(b"mflowy").hexdigest(),
        "size_bytes": 6,
    }


def test_file_hash_stable_fingerprint(tmp_path):
    """工具定位=变更检查：sha256 单算法，内容变则指纹变"""
    f = tmp_path / "data.bin"
    f.write_bytes(b"abc")
    h1 = file_hash(str(f))
    assert h1["sha256"] == hashlib.sha256(b"abc").hexdigest()
    f.write_bytes(b"abd")
    assert file_hash(str(f))["sha256"] != h1["sha256"]


def test_file_hash_streams_chunks(tmp_path):
    """超过分块边界（8MB）的文件走多块路径，size/哈希仍准确"""
    f = tmp_path / "big.bin"
    payload = b"\0" * (8 * 1024 * 1024 + 1)
    f.write_bytes(payload)
    result = file_hash(str(f))
    assert result["size_bytes"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()


def test_file_hash_missing_file_returns_error():
    assert file_hash("/no/such/file.csv").startswith("Error: File not found")


def test_file_hash_directory_returns_error(tmp_path):
    assert file_hash(str(tmp_path)).startswith("Error: Path is not a file")


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


def test_list_modules_annotates_builtin_requires():
    """内置模块带 requires 标注（base 环境可见装哪个 extra 可用）"""
    entries = list_modules(step="model")
    assert entries[0]["requires"]["XGB"] == "modeling"
    # data_analysis 图表属 stats（目录边界）
    plot = list_modules(step="plot")
    requires = plot[0]["requires"]
    assert requires["correlation_heatmap"] == "stats"
    assert requires["shap_summary"] == "modeling"  # plots 二级目录细分


def test_get_module_info_unavailable_reports_extra(monkeypatch):
    """缺 extra 环境加载失败 → 结构化 available=false + 所需 extra，而非裸 ImportError"""
    import mflowy.driver.discover as discover_mod

    def _missing_dep(step, module):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(discover_mod, "_load_fn", _missing_dep)
    info = get_module_info("model", "XGB")
    assert info["available"] is False
    assert info["requires"] == "modeling"
    assert "[modeling]" in info["reason"]
