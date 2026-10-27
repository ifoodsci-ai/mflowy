"""会话基座机制：SessionStore（目录/manifest IO、终态、弃局、GC）。

driver=机制、域=策略——动作/检测/质量/守卫/版本树全部留域；本模块不读 manifest
业务字段（终态双键与可选 ``steps``/``abandon_reason`` 除外）。终态不变式：
``finished``/``abandoned`` 互斥，活会话双缺省（读取向后兼容——存量 manifest 无
abandoned 键按 False）。
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from mflowy.utils.file import write_text_atomic


def gc_report_key(dry_run: bool) -> str:
    """gc 报告键随 dry_run 切换：would_delete（只列）/ deleted（已删）——键名单一来源。"""
    return "would_delete" if dry_run else "deleted"


def is_terminal_state(manifest: Mapping[str, Any]) -> bool:
    """manifest 级终态判定（共享单源，M3）：``finished ∨ abandoned``，活会话双缺省。

    域会话的终态判定应委托本函数（消两域逐字重复）；
    SessionStore 内部消费方（``is_terminal``/``abandon``）同样走此单源。
    """
    return bool(manifest.get("finished", False) or manifest.get("abandoned", False))


class SessionStore:
    """会话生命周期机制：目录/manifest IO、终态、弃局、GC（root 由域注入）。

    driver=机制、域=策略——动作/检测/质量/守卫/版本树全部留域；本类不读 manifest
    业务字段（终态双键与可选 ``steps``/``abandon_reason`` 除外）。终态不变式：
    ``finished``/``abandoned`` 互斥，活会话双缺省（读取向后兼容——存量 manifest 无
    abandoned 键按 False）。
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def create(self, sid: str, manifest: Mapping[str, Any]) -> Path:
        """开局：建会话目录并落初版 manifest（sid 已存在即 FileExistsError）。"""
        self.root.mkdir(parents=True, exist_ok=True)
        d = self.root / sid
        d.mkdir(exist_ok=False)
        self.write_manifest(sid, manifest)
        return d

    def session_dir(self, sid: str) -> Path:
        d = self.root / sid
        if not d.is_dir():
            raise FileNotFoundError(f"会话不存在: {sid}")
        return d

    def read_manifest(self, sid: str) -> dict[str, Any]:
        return json.loads(self.session_dir(sid).joinpath("manifest.json").read_text(encoding="utf-8"))

    def write_manifest(self, sid: str, manifest: Mapping[str, Any]) -> None:
        """原子落盘（tmp+rename，M3）：manifest 是会话 commit 点，撕裂写不留半开状态。"""
        write_text_atomic(
            self.session_dir(sid) / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=1, default=str),
        )

    def is_terminal(self, sid: str) -> bool:
        """终态判定：finished ∨ abandoned（get 缺省——存量会话文件零迁移）。"""
        return is_terminal_state(self.read_manifest(sid))

    def abandon(self, sid: str, reason: str | None = None) -> dict[str, Any]:
        """弃局终态化：置 abandoned、reason 落盘（审计）；保留决策日志与版本树、
        不导出交付物；已终态再弃即 raise。

        ack = {session_id, abandoned, reason, steps}——steps 取 manifest 的
        ``steps`` 键（缺省 0；域迁移时显式落盘），记"弃在第几步"。
        """
        m = self.read_manifest(sid)
        if is_terminal_state(m):
            raise ValueError(f"会话已终态: {sid}")
        m["abandoned"] = True
        if reason is not None:
            m["abandon_reason"] = reason
        self.write_manifest(sid, m)
        return {"session_id": sid, "abandoned": True, "reason": reason, "steps": m.get("steps", 0)}

    def gc(self, ttl_days: int = 30, dry_run: bool = True) -> dict[str, list]:
        """状态无关回收：本 root 内按 TTL 清理会话目录（活/死/损坏一律按龄）。

        不读 manifest 内容——损坏/不可读会话照样入删单（GC 的鲁棒性卖点）；
        mtime 锚 manifest.json（目录 mtime 会被版本工件写入扰动；无 manifest 的
        目录回退自身 mtime）。dry_run=True 只列不删（键 ``would_delete``；执行时
        键 ``deleted``）；root 内非目录条目不动作（``ignored``）。root 不存在 =
        无可回收，返回空报告（显式 root 参数路径无前置过滤，守卫在基座）。
        """
        entries: list[dict[str, Any]] = []
        ignored: list[str] = []
        if not self.root.is_dir():
            return {gc_report_key(dry_run): entries, "ignored": ignored}
        now = time.time()
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                ignored.append(str(child))
                continue
            anchor = child / "manifest.json"
            mtime = anchor.stat().st_mtime if anchor.exists() else child.stat().st_mtime
            age_days = (now - mtime) / 86400
            if age_days < ttl_days:  # TTL 边界：恰满 ttl_days 即过期
                continue
            entries.append(
                {
                    "root": str(self.root),
                    "session_id": child.name,
                    "last_modified": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
                    "age_days": round(age_days, 2),
                }
            )
            if not dry_run:
                shutil.rmtree(child)
        return {gc_report_key(dry_run): entries, "ignored": ignored}
