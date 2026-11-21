"""遥测开关仲裁（#243 TTY 分流）：env MFLOWY_TELEMETRY > settings.json telemetry（bool）
> 未决（TTY 首问落盘 / 非 TTY 默认开 + stderr 指引）。

settings.json 固定读 ~/.mflowy/settings.json（HOME 经 monkeypatch 隔离）。
"""

import json
from pathlib import Path

import pytest
from mflowy.app import telemetry

ENV_MODE = telemetry.ENV_MODE


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """HOME 指向临时目录保证「未设」态密闭（新实现无 @cache，零缓存清理面）。"""
    monkeypatch.setenv("HOME", str(tmp_path))


def _write_settings(tmp_path, value):
    if value is None:
        return
    path = Path(tmp_path) / ".mflowy" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"telemetry": value}), encoding="utf-8")


@pytest.mark.parametrize(
    ("env", "file_value", "expected"),
    [
        ({"MFLOWY_TELEMETRY": "on"}, True, True),
        ({"MFLOWY_TELEMETRY": "on"}, False, True),  # env 显式 on 压过文件 off（运维逃生舱）
        ({"MFLOWY_TELEMETRY": "off"}, True, False),
        ({}, True, True),  # env 未设 → 查文件
        ({}, False, False),
        ({}, None, True),  # 无值 + 非 TTY（pytest）→ 默认开
        ({"MFLOWY_TELEMETRY": "ask"}, True, True),  # ask = 显式"查文件，空则分流"
        ({"MFLOWY_TELEMETRY": "ask"}, None, True),  # 非 TTY ask → 默认开
        ({"MFLOWY_TELEMETRY": "banana"}, True, True),  # 非法值按 ask
        ({"MFLOWY_TELEMETRY": "ON"}, None, True),  # 大小写敏感，ON 非法 → 按 ask
    ],
)
def test_resolve_env_over_file(env, file_value, expected, tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_MODE, raising=False)  # 清宿主/conftest 环境，保证「未设」态密闭
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _write_settings(tmp_path, file_value)
    assert telemetry.resolve_telemetry() is expected


def test_write_telemetry_persists_and_resolves(tmp_path, monkeypatch):
    """同意结果（on/off 均持久化）写入后直接生效、不再未决。"""
    monkeypatch.delenv(ENV_MODE, raising=False)
    telemetry.write_telemetry(True)
    assert telemetry.resolve_telemetry() is True
    telemetry.write_telemetry(False)
    assert telemetry.resolve_telemetry() is False


def test_write_telemetry_preserves_other_keys(tmp_path):
    """目录不存在自建；已存在的无关配置键不丢。"""
    settings = Path(tmp_path) / ".mflowy" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"other": {"lang": "zh"}}), encoding="utf-8")
    telemetry.write_telemetry(True)
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["other"] == {"lang": "zh"}
    assert data["telemetry"] is True


def test_endpoint_defaults_to_official(monkeypatch):
    """env 未设 → 官方默认端点。"""
    monkeypatch.delenv(telemetry.ENV_ENDPOINT, raising=False)
    assert telemetry.telemetry_endpoint() == telemetry.DEFAULT_ENDPOINT


def test_endpoint_env_overrides_for_self_hosted_gateway(monkeypatch):
    """env 指向自建网关（docker/telemetry）——注册与导出共用的端点解析可逃生。"""
    monkeypatch.setenv(telemetry.ENV_ENDPOINT, "http://127.0.0.1:8080")
    assert telemetry.telemetry_endpoint() == "http://127.0.0.1:8080"
