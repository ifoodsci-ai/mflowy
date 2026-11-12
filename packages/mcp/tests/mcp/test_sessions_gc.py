"""sessions_gc 平台工具：约定扫描（当前任务锚 anchor/*_sessions，#207）+ 薄委托 driver SessionStore.gc。

分层裁定：driver=机制（按 root 清理），mcp=平台约定（哪些 root）——driver 对用户
home 零知识。审计面：dry_run 只列不删（安全缺省），执行后回 deleted 清单。
缺省扫描面 = task_dir 锚（无锚落 ~/.mflowy——旧扫描路径逐字相同）；显式 root
参数保留救场（跨锚回收旧根）。
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

    def test_bound_anchor_scans_under_task_dir(self, tmp_path):
        """task_dir 绑定（钉住/锚参数推导同机制）时缺省扫 <anchor>/*_sessions——
        task_dir 下会话根可被回收（#207 缺省扫描面对齐）。"""
        from mflowy.mcp.tools import sessions_gc
        from mflowy.utils.path import pin_task_dir, reset_task_dir

        anchor = tmp_path / "workspace"
        _mk_session(anchor / "clean_sessions", "s1", 40)
        _mk_session(anchor / "unrelated", "s2", 40)  # 不匹配 *_sessions 约定，不入扫描
        token = pin_task_dir(str(anchor))
        try:
            report = sessions_gc()
        finally:
            reset_task_dir(token)
        assert report["scanned_roots"] == [str(anchor / "clean_sessions")]
        assert [e["session_id"] for e in report["would_delete"]] == ["s1"]
        assert (anchor / "clean_sessions" / "s1").is_dir()
