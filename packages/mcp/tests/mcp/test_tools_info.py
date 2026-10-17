"""info 工具行为测试 — file_hash / list_nodes / get_node_info 返回 JSON 消息。"""

import hashlib

import pytest
from mflowy.mcp.tools import file_hash, get_node_info, list_nodes


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


def test_file_hash_missing_file_raises():
    """错误契约：路径错误抛异常（runner/MCP 层标准错误通道），不返错误字符串（validate 例外——校验失败是其领域结果，返回 valid=False）"""
    import pytest

    with pytest.raises(FileNotFoundError, match="文件不存在"):
        file_hash("/no/such/file.csv")


def test_file_hash_directory_raises(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="路径不是文件"):
        file_hash(str(tmp_path))


def test_list_nodes_groups():
    data = list_nodes()
    steps = {item["namespace"] for item in data}
    assert "model" in steps and "load" in steps
    model = next(i for i in data if i["namespace"] == "model")
    # 节点条目结构化 {name=node_key, upstream}——边消费扁平展开为 via 标注（#59）
    xgb = next(e for e in model["nodes"] if e["name"] == "model__XGB")
    assert xgb["upstream"][0]["name"] == "x_y"
    assert xgb["upstream"][0]["via"] == "train_split"


def test_list_nodes_filter_by_namespace():
    data = list_nodes(namespace="model")
    assert [i["namespace"] for i in data] == ["model"]


def test_list_nodes_invalid_namespace_lists_valid_options():
    with pytest.raises(ModuleNotFoundError, match="nope"):
        list_nodes(namespace="nope")


def test_get_node_info_happy_path():
    info = get_node_info("model", "XGB")
    assert info["key"] == "model__XGB"  # node_key：装饰器声明身份（`__` 连接符）
    assert "XGBoost" in info["description"]  # doc 首行
    schema = info["parameters_schema"]
    props = schema["properties"]
    assert "n_estimators" in props and "scoring" in props
    # 注入参数（裸类型）不在用户可配 schema 中
    assert "task" not in props and "dataset_loader" not in props
    n_trials = props["n_trials"]
    assert n_trials["type"] == "integer"
    assert n_trials["default"] == 100
    assert "优化试验次数" in n_trials["description"]  # 旧式字符串 Annotated 桥接为 description
    # 搜索空间默认实例编码为结构化 JSON（dataclass→对象 / list 子类→数组）
    assert props["max_depth"]["default"] == {"start": 3, "end": 10, "step": 1}


def test_get_node_info_required_param_in_schema():
    """必填参数进 required 数组（JSON Schema 标准表达）"""
    info = get_node_info("load", "csv")
    schema = info["parameters_schema"]
    assert "source" in schema["required"]
    assert "default" not in schema["properties"]["source"]


def test_get_node_info_unknown_module():
    """词表 miss（节点名打错）透传 CatalogMiss + Available，不进 extras 降级分支。"""
    from mflowy.driver.discover import CatalogMiss

    with pytest.raises(CatalogMiss, match="Available"):
        get_node_info("model", "NoSuchModule")


def test_list_nodes_annotates_builtin_requires():
    """0.2.0 extras 语义：重栈为域包硬依赖零标注，唯一可选 extra=[deep]（torch/MLP）"""
    entries = list_nodes(namespace="model")
    requires = entries[0]["requires"]
    assert "XGB" not in requires  # CPU 建模栈硬依赖——零标注即裸装可用
    assert requires.get("MLP") == "deep"  # torch 栈隔离进 [deep]
    # 数据栈/解释栈均为硬依赖：plot 族零标注
    plot = list_nodes(namespace="plot")
    assert not plot[0].get("requires")


def test_get_node_info_unavailable_reports_extra(monkeypatch):
    """缺 extra 环境加载失败 → 结构化 available=false + 所需 extra，而非裸 ImportError"""
    import mflowy.driver.discover as discover_mod

    def _missing_dep(key, valid="handler"):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(discover_mod, "_load_fn", _missing_dep)
    info = get_node_info("model", "MLP")
    assert info["available"] is False
    assert info["requires"] == "deep"
    assert "[deep]" in info["reason"]
