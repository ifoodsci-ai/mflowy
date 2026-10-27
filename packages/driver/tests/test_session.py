"""session 会话基座：POMDPSession 壳住 test_pomdp_session.py；本文件 = SessionStore
GC 面（manifest.json 随 #89 溶解，生命周期测试随面退役——壳测试见 test_pomdp_session）。
"""

import os
import time

from mflowy.driver.session import SessionStore


def _mk_session(root, sid: str, with_observation: bool = True) -> None:
    """直建会话目录（gc 只看 mtime——observation.json 锚 #89 新格式，manifest.json 锚存量）。"""
    d = root / sid
    d.mkdir(parents=True)
    (d / ("observation.json" if with_observation else "manifest.json")).write_text("{}", encoding="utf-8")


class TestGc:
    @staticmethod
    def _age(store_root, sid, days):
        """把会话的锚文件 mtime 拨到 days 天前（TTL 输入的独立真源）。"""
        stamp = time.time() - days * 86400
        d = store_root / sid
        anchor = next((d / n for n in ("observation.json", "manifest.json") if (d / n).exists()), None)
        target = anchor if anchor is not None else d
        os.utime(target, (stamp, stamp))

    def test_ttl_dry_run_and_delete(self, tmp_path):
        """恰满 ttl_days 即过期；dry_run 只列不删，执行真删；fresh 会话保留。"""
        store = SessionStore(tmp_path)
        for sid in ("old-exact", "old-er", "fresh"):
            _mk_session(tmp_path, sid)
        self._age(tmp_path, "old-exact", days=30)  # TTL 边界：恰满 30 天
        self._age(tmp_path, "old-er", days=31)
        self._age(tmp_path, "fresh", days=1)

        report = store.gc(ttl_days=30, dry_run=True)  # 缺省 dry_run 只列不删
        assert {e["session_id"] for e in report["would_delete"]} == {"old-exact", "old-er"}
        entry = report["would_delete"][0]
        assert entry["root"] == str(tmp_path) and entry["age_days"] >= 30 and entry["last_modified"]
        assert (tmp_path / "old-exact" / "observation.json").exists()  # 未删

        report = store.gc(ttl_days=30, dry_run=False)
        assert {e["session_id"] for e in report["deleted"]} == {"old-exact", "old-er"}
        assert not (tmp_path / "old-exact").exists() and not (tmp_path / "old-er").exists()
        assert (tmp_path / "fresh").is_dir()  # fresh 保留

    def test_corrupt_sessions_gcable(self, tmp_path):
        """损坏会话照样入删单：GC 不读会话内容（坏 JSON / 无锚文件均按 mtime 回收）。"""
        store = SessionStore(tmp_path)
        (tmp_path / "corrupt-json").mkdir()
        (tmp_path / "corrupt-json" / "observation.json").write_text("{not-json", encoding="utf-8")
        (tmp_path / "no-anchor").mkdir()
        stamp = time.time() - 40 * 86400
        for sid in ("corrupt-json", "no-anchor"):
            os.utime(tmp_path / sid, (stamp, stamp))  # 无锚文件：回退目录 mtime
            for anchor in ("observation.json", "manifest.json"):
                if (f := tmp_path / sid / anchor).exists():
                    os.utime(f, (stamp, stamp))

        report = store.gc(ttl_days=30, dry_run=True)
        assert {e["session_id"] for e in report["would_delete"]} == {"corrupt-json", "no-anchor"}

    def test_ignores_non_dirs(self, tmp_path):
        """root 内非目录条目不动作（ignored 清单），不误删旁边文件。"""
        store = SessionStore(tmp_path)
        (tmp_path / "README.txt").write_text("stray", encoding="utf-8")

        report = store.gc(ttl_days=0, dry_run=True)
        assert report["ignored"] == [str(tmp_path / "README.txt")]
        assert (tmp_path / "README.txt").exists()

    def test_missing_root_returns_empty_report(self, tmp_path):
        """root 不存在 = 无可回收：空报告（不炸 FileNotFoundError；等价 mcp 侧 is_dir 前置过滤）。"""
        store = SessionStore(tmp_path / "never-created")
        assert store.gc(ttl_days=30, dry_run=True) == {"would_delete": [], "ignored": []}
        assert store.gc(ttl_days=30, dry_run=False) == {"deleted": [], "ignored": []}
