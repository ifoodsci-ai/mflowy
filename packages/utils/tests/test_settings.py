"""全局配置根 settings.json 契约：键值读写单源（tmp+rename 原子、保留无关键）。"""

import json

from mflowy.utils import settings


def test_read_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert settings.read_settings() == {}
    assert settings.read_key("telemetry") is None


def test_read_bad_json_returns_empty(tmp_path, monkeypatch):
    """坏 JSON / 非对象顶层一律按未配置处理（返回空，不抛）。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".mflowy" / "settings.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{not json", encoding="utf-8")
    assert settings.read_settings() == {}
    cfg.write_text('["a list"]', encoding="utf-8")
    assert settings.read_settings() == {}


def test_write_key_roundtrip_and_directory_autocreate(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    settings.write_key("job_provider", "remote")
    path = tmp_path / ".mflowy" / "settings.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"job_provider": "remote"}
    assert settings.read_key("job_provider") == "remote"


def test_write_key_preserves_unrelated_keys(tmp_path, monkeypatch):
    """写单键不丢无关配置键（telemetry 与 job_provider 共居一文件——#243 两消费方）。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    settings.write_key("telemetry", True)
    settings.write_key("job_provider", "local")
    data = json.loads((tmp_path / ".mflowy" / "settings.json").read_text(encoding="utf-8"))
    assert data == {"telemetry": True, "job_provider": "local"}


def test_write_key_leaves_no_temp_residue(tmp_path, monkeypatch):
    """tmp+rename 原子落盘：成功路径不留 .settings.* 残片。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    settings.write_key("k", 1)
    residues = list((tmp_path / ".mflowy").glob(".settings.*"))
    assert residues == []
