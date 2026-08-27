"""遥测开关仲裁：env MFLOWY_TELEMETRY > settings.json telemetry（bool）> 未决（ask）。

settings.json 固定读 ~/.mflowy/settings.json（HOME 经 monkeypatch 隔离）；env 经
monkeypatch 注入（自动还原 + 隔离宿主机环境变量），矩阵用例的「未设」态先 delenv 清场。
"""

import json
from pathlib import Path

import pytest
from mflowy.mcp import telemetry

ENV_MODE = telemetry.ENV_MODE


@pytest.fixture(autouse=True)
def _isolate_telemetry_state(tmp_path, monkeypatch):
    """settings_path/api_key_path/_read_api_key 为 @cache，跨用例缓存宿主机路径——
    每用例清 cache 并把 HOME 指向临时目录，保证「未设」态密闭。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    telemetry.settings_path.cache_clear()
    telemetry.api_key_path.cache_clear()
    telemetry._read_api_key.cache_clear()
    yield
    telemetry.settings_path.cache_clear()
    telemetry.api_key_path.cache_clear()
    telemetry._read_api_key.cache_clear()


def _write_settings(tmp_path, value):
    if value is None:
        return
    path = Path(tmp_path) / ".mflowy" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"telemetry": value}), encoding="utf-8")


@pytest.mark.parametrize(
    ("env", "file_value", "expected"),
    [
        ({"MFLOWY_TELEMETRY": "on"}, True, "on"),
        ({"MFLOWY_TELEMETRY": "on"}, False, "on"),  # env 显式 on 压过文件 off（运维逃生舱）
        ({"MFLOWY_TELEMETRY": "off"}, True, "off"),
        ({}, True, "on"),  # env 未设 → 查文件
        ({}, False, "off"),
        ({}, None, "ask"),  # 无值 → 未决，触发 elicitation
        ({"MFLOWY_TELEMETRY": "ask"}, True, "on"),  # ask = 显式"查文件，空则问"
        ({"MFLOWY_TELEMETRY": "ask"}, None, "ask"),
        ({"MFLOWY_TELEMETRY": "banana"}, True, "on"),  # 非法值按 ask
        ({"MFLOWY_TELEMETRY": "banana"}, None, "ask"),
        ({"MFLOWY_TELEMETRY": "ON"}, None, "ask"),  # 大小写敏感，ON 非法
    ],
)
def test_resolve_env_over_file(env, file_value, expected, tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_MODE, raising=False)  # 清宿主环境，保证「未设」态密闭
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _write_settings(tmp_path, file_value)
    assert telemetry.resolve_telemetry() == expected


def test_write_telemetry_persists_and_resolves(tmp_path, monkeypatch):
    """同意结果（on/off 均持久化）写入后，ask 模式直接生效、不再未决。"""
    monkeypatch.delenv(ENV_MODE, raising=False)
    telemetry.write_telemetry(True)
    assert telemetry.resolve_telemetry() == "on"
    telemetry.write_telemetry(False)
    assert telemetry.resolve_telemetry() == "off"


def test_write_telemetry_preserves_other_keys(tmp_path):
    """目录不存在自建；已存在的无关配置键不丢。"""
    settings = Path(tmp_path) / ".mflowy" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"other": {"lang": "zh"}}), encoding="utf-8")
    telemetry.write_telemetry(True)
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["other"] == {"lang": "zh"}
    assert data["telemetry"] is True
