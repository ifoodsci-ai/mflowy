"""info 工具行为测试 — list_modules / get_module_info 返回 JSON 消息。"""

import pytest

from mflowy.mcp.tools import get_module_info, list_modules


def test_list_modules_groups():
    data = list_modules()
    steps = {item["step"] for item in data}
    assert "model" in steps and "load" in steps
    model = next(i for i in data if i["step"] == "model")
    assert "XGB" in model["modules"]


def test_list_modules_filter_by_step():
    data = list_modules(step="model")
    assert [i["step"] for i in data] == ["model"]


def test_list_modules_step_accepts_name_form():
    """枚举名（MODEL）与枚举值（model）都应命中"""
    assert [i["step"] for i in list_modules(step="MODEL")] == ["model"]
    assert [i["step"] for i in list_modules(step="X_y")] == ["X_y"]
    assert [i["step"] for i in list_modules(step="XY")] == ["X_y"]


def test_list_modules_invalid_step_lists_valid_options():
    with pytest.raises(ValueError, match="StepType"):
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
