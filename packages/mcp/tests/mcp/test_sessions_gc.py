"""sessions_gc 平台工具：约定扫描（~/.mflowy/*_sessions）+ 薄委托 driver SessionStore.gc。

分层裁定：driver=机制（按 root 清理），mcp=平台约定（哪些 root）——driver 对用户
home 零知识。审计面：dry_run 只列不删（安全缺省），执行后回 deleted 清单。
"""

import os
import time
from pathlib import Path


def _mk_session(root: Path, sid: str, days_old: float) -> None:
    d = root / sid
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}", encoding="utf-8")
    stamp = time.time() - days_old * 86400
    os.utime(d / "manifest.json", (stamp, stamp))


class TestExplicitRoot:
    def test_default_dry_run_lists_without_deleting(self, tmp_path):
        from mflowy.mcp.tools import sessions_gc

        _mk_session(tmp_path, "old", 40)
        report = sessions_gc(root=str(tmp_path))
        assert report["scanned_roots"] == [str(tmp_path)]
        assert [e["session_id"] for e in report["would_delete"]] == ["old"]
        assert (tmp_path / "old").is_dir()  # 缺省只列不删

    def test_execute_deletes_and_keeps_fresh(self, tmp_path):
        from mflowy.mcp.tools import sessions_gc

        _mk_session(tmp_path, "old", 40)
        _mk_session(tmp_path, "fresh", 1)
        report = sessions_gc(root=str(tmp_path), ttl_days=30, dry_run=False)
        assert [e["session_id"] for e in report["deleted"]] == ["old"]
        assert not (tmp_path / "old").exists()
        assert (tmp_path / "fresh").is_dir()


class TestConventionScan:
    def test_discovers_sessions_roots_under_home(self, tmp_path, monkeypatch):
        """缺省扫描 ~/.mflowy/*_sessions：全部约定 root 的过期会话聚合出账。"""
        from mflowy.mcp.tools import sessions_gc

        home = tmp_path / "home"
        base = home / ".mflowy"
        for name in ("clean_sessions", "modeling_sessions", "misc_dir"):
            _mk_session(base / name, "s1", 40)  # misc_dir 不匹配约定，不入扫描
        monkeypatch.setattr(Path, "home", lambda: home)

        report = sessions_gc()
        assert sorted(report["scanned_roots"]) == [
            str(base / "clean_sessions"),
            str(base / "modeling_sessions"),
        ]
        assert {e["session_id"] for e in report["would_delete"]} == {"s1"}
        assert all((base / n / "s1").is_dir() for n in ("clean_sessions", "modeling_sessions"))

    def test_no_roots_found_returns_empty_report(self, tmp_path, monkeypatch):
        from mflowy.mcp.tools import sessions_gc

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        report = sessions_gc()
        assert report["scanned_roots"] == [] and report["would_delete"] == [] and report["ignored"] == []
