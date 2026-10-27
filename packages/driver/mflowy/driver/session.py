"""会话基座机制：SessionStore（存量 manifest IO）+ POMDPSession（会话壳，#89）。

driver=机制、域=策略——POMDPSession 吃下壳骨架：六个操作上下文承载业务流程
（终态门/预算门/prior 冻结/append-only 决策日志/统一盘上写序），域实现
``pomdp.POMDPSessionHandle`` 只填转移语义。盘上三文件：

- ``observation.json``：观测摘要（七键+终态双旗，pydantic 封闭模型），整读改写、
  原子落盘——观测投影，最后写
- ``decision.log``：决策日志（JSONL append-only，一行一条 (s,a,r,s′)）——
  ``ra`` 接口无删除面；**追加即 commit 点**
- ``rules.pkl``：规则引擎实例（整读整写、异常零副作用——语义沿 rule 模块）

统一写序（构造保证）：工件先行（域、未 commit、孤儿无害）→ 日志追加（commit）→
观测投影最后落盘。崩溃窗口二分：追加前崩 = 失败免费；追加后投影前崩 = 日志领先，
``open`` 的一致性牙齿（recent_decisions 尾 ≡ 日志尾）fail-loud。
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from mflowy.utils.file import write_text_atomic

from .pomdp import TOP_K_DECISIONS, Observation
from .rule import Rule, RuleEngine, load_rule_engine, save_rule_engine

OBSERVATION_FILE = "observation.json"
DECISION_LOG_FILE = "decision.log"
RULES_FILE = "rules.pkl"


def gc_report_key(dry_run: bool) -> str:
    """gc 报告键随 dry_run 切换：would_delete（只列）/ deleted（已删）——键名单一来源。"""
    return "would_delete" if dry_run else "deleted"


def is_terminal_state(manifest: Mapping[str, Any]) -> bool:
    """manifest 级终态判定（共享单源，M3）：``finished ∨ abandoned``，活会话双缺省。

    SessionStore（存量 manifest 面向）内部消费方走此单源；POMDPSession 走
    Observation 终态双旗（同一谓词的模型面）。
    """
    return bool(manifest.get("finished", False) or manifest.get("abandoned", False))


class SessionStore:
    """会话生命周期机制（存量 manifest 面）：目录/manifest IO、终态、弃局、GC。

    driver=机制、域=策略；本类不读 manifest 业务字段（终态双键与可选
    ``steps``/``abandon_reason`` 除外）。终态不变式：``finished``/``abandoned``
    互斥，活会话双缺省（读取向后兼容——存量 manifest 无 abandoned 键按 False）。
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

        不读会话内容——损坏/不可读会话照样入删单（GC 的鲁棒性卖点）；
        mtime 锚 observation.json（#89 新格式），回退 manifest.json（存量）、
        再回退目录自身 mtime。dry_run=True 只列不删（键 ``would_delete``；
        执行时键 ``deleted``）；root 内非目录条目不动作（``ignored``）。
        root 不存在 = 无可回收，返回空报告。
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
            anchor = next((child / n for n in (OBSERVATION_FILE, "manifest.json") if (child / n).exists()), None)
            mtime = anchor.stat().st_mtime if anchor else child.stat().st_mtime
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


class _DecisionLog:
    """决策日志门面：append 写穿（= commit 点）+ 只读历史（无删除面）。

    只读重入态（finish 幂等重入）下 append 即 raise——双 commit 由构造杜绝。
    """

    def __init__(self, path: Path, *, readonly: bool = False):
        self._path = path
        self._readonly = readonly
        self._records: list[dict] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    def append(self, record: Mapping[str, Any]) -> None:
        """写穿：整行单次 write 追加（commit）——异常上下文中已追加的记录保留，
        由 open 一致性牙齿兜底（日志领先投影即 fail-loud）。
        """
        if self._readonly:
            raise RuntimeError("会话已收局：只读重入（幂等重导出），决策日志不可追加")
        line = json.dumps(record, ensure_ascii=False, default=str)
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self._records.append(dict(record))

    def tail(self, n: int) -> list[Mapping[str, Any]]:
        return list(self._records[-n:])

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[dict]:
        return iter(self._records)

    def __getitem__(self, i: int) -> dict:
        return self._records[i]


class POMDPSession:
    """POMDP 会话壳（#89 统一机制）：所有业务共用，域只填转移语义。

    新开 = ``POMDPSession(root, *rules, prefix)``：mint id + 建目录 + 规则引擎
    由 ``*rules`` 现建落盘 + 空决策日志；开局观测由 ``start_ctx`` 提交。
    重水化 = ``POMDPSession.open(dir)``：入口一致性牙齿（观测投影 ≡ 日志尾）。
    步数口径：``counts_step(action) -> bool`` 模块级谓词（可 pickle），缺省
    None = 每 apply 计步；预算门 = counts_step(action) ∧ steps ≥ max_steps。
    """

    def __init__(
        self,
        root: Path | str,
        *rules: Rule,
        prefix: str = "",
        counts_step: Callable[[str], bool] | None = None,
    ) -> None:
        self._root = Path(root)
        self._counts_step = counts_step
        sid = f"{prefix}{uuid.uuid4().hex[:12]}"
        self._root.mkdir(parents=True, exist_ok=True)
        self._dir = self._root / sid
        self._dir.mkdir(exist_ok=False)  # sid 冲突 fail-loud
        self._id = sid
        save_rule_engine(str(self._dir / RULES_FILE), RuleEngine().register(*rules))
        (self._dir / DECISION_LOG_FILE).write_text("", encoding="utf-8")

    @classmethod
    def open(cls, dir: Path | str, *, counts_step: Callable[[str], bool] | None = None) -> POMDPSession:
        """重水化既有会话：目录/开局观测缺失 fail-loud；投影与日志撕裂 fail-loud。"""
        ss = cls.__new__(cls)
        ss._root = Path(dir).parent
        ss._dir = Path(dir)
        if not ss._dir.is_dir():
            raise FileNotFoundError(f"会话不存在: {dir}")
        obs_path = ss._dir / OBSERVATION_FILE
        if not obs_path.exists():
            raise FileNotFoundError(f"会话未开局（缺 observation.json）: {dir}")
        ss._counts_step = counts_step
        ss._id = ss._dir.name
        obs = ss._load_observation()
        if list(obs.recent_decisions) != _DecisionLog(ss._dir / DECISION_LOG_FILE).tail(TOP_K_DECISIONS):
            raise RuntimeError(f"观测投影与决策日志不一致（撕裂写嫌疑，日志领先）: {dir}")
        return ss

    @property
    def id(self) -> str:
        return self._id

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def is_terminal(self) -> bool:
        o = self._load_observation()
        return o.finished or o.abandoned

    @property
    def observation(self) -> Observation:
        """只读观测视图（preview/终局计算等零副作用读；可变面走操作上下文）。"""
        return self._load_observation()

    @property
    def decision_log(self) -> _DecisionLog:
        """只读决策日志（append 即 raise——读历史用）。"""
        return _DecisionLog(self._dir / DECISION_LOG_FILE, readonly=True)

    # ── 六操作上下文（公共 interface）───────────────────────────────────────

    @contextmanager
    def start_ctx(self, max_steps: int, *, prior: Mapping[str, Any] | None = None, state_digest: Mapping[str, Any] | None = None) -> Iterator[Observation]:
        """开局：body 内物化工件（工件先行），exit 提交观测（开局 commit 点）。"""
        if (self._dir / OBSERVATION_FILE).exists():
            raise FileExistsError(f"会话已开局: {self._id}")
        o = Observation(session_id=self._id, max_steps=max_steps, prior=prior, state_digest=state_digest or {})
        yield o
        self._write_observation(o)

    @contextmanager
    def propose_ctx(self) -> Iterator[tuple[Observation, RuleEngine]]:
        """出题：可变观测 + 规则引擎（检测证据进 state_digest、fire(o)/poll 在 body）。

        终态门兜底 raise——域应先查 ``is_terminal`` 返回空菜单（建议通道）。
        exit 写序：规则引擎 → 观测投影最后。
        """
        o = self._gate_mutable()
        engine = self._load_engine()
        yield o, engine
        save_rule_engine(str(self._dir / RULES_FILE), engine)
        self._write_observation(o)

    @contextmanager
    def apply_ctx(self, action: str) -> Iterator[tuple[Observation, _DecisionLog]]:
        """执行动作：终态门 + 预算门；``l.append`` 写穿即 commit；exit 投影最后。

        body 内失败即免费（追加前抛出：零日志、零观测变更、孤儿工件无害）。
        """
        o = self._gate_mutable()
        if self._action_counts(action) and o.steps >= o.max_steps:
            raise RuntimeError(f"预算耗尽（max_steps={o.max_steps}）：请 rollback 或 finish")
        log = _DecisionLog(self._dir / DECISION_LOG_FILE)
        yield o, log
        self._write_observation(o)

    @contextmanager
    def rollback_ctx(self) -> Iterator[tuple[Observation, _DecisionLog]]:
        """回滚：域在 body 定位目标记录、恢复指针状态、追加 rollback 记录（不删行）。"""
        o = self._gate_mutable()
        log = _DecisionLog(self._dir / DECISION_LOG_FILE)
        yield o, log
        self._write_observation(o)

    @contextmanager
    def finish_ctx(self) -> Iterator[tuple[Observation, _DecisionLog]]:
        """收局：abandoned → raise（改协议重开走新会话）；finished → 只读重入
        （``l.append`` 即 raise、exit 不落盘——幂等重导出由构造保证，交付物
        工件写不受门管）；活会话 → body 置 ``o.finished=True``、追加 finish
        记录，exit 投影（拒收路不置 finished，会话保留可续局）。
        """
        o = self._load_observation()
        if o.abandoned:
            raise RuntimeError(f"会话已弃局：改协议重开走新会话（{self._id}）")
        if o.finished:
            yield o, _DecisionLog(self._dir / DECISION_LOG_FILE, readonly=True)
            return  # 幂等重入：零落盘
        log = _DecisionLog(self._dir / DECISION_LOG_FILE)
        yield o, log
        self._write_observation(o)

    def abandon(self, reason: str | None = None) -> dict[str, Any]:
        """弃局：日志记录 + 观测置 abandoned。幂等（已弃局重复弃 = no-op 回执）；
        已收局不可弃（二终态互斥）。
        """
        o = self._load_observation()
        if o.finished:
            raise RuntimeError(f"会话已终态（收局）：不可弃局（{self._id}）")
        if o.abandoned:
            return {"session_id": self._id, "abandoned": True, "reason": reason, "steps": o.steps}
        o.abandoned = True
        _DecisionLog(self._dir / DECISION_LOG_FILE).append({"action": "abandon", "reason": reason})
        self._write_observation(o)
        return {"session_id": self._id, "abandoned": True, "reason": reason, "steps": o.steps}

    # ── 内部件（三个 RWA 原语与门）─────────────────────────────────────────

    def _gate_mutable(self) -> Observation:
        """可变打开的终态门：已终态会话动作面为空（apply/rollback/propose 共用）。"""
        o = self._load_observation()
        if o.finished or o.abandoned:
            raise RuntimeError(f"会话已终态（收局/弃局）：动作面为空（{self._id}）")
        return o

    def _action_counts(self, action: str) -> bool:
        return self._counts_step is None or bool(self._counts_step(action))

    def _load_observation(self) -> Observation:
        p = self._dir / OBSERVATION_FILE
        if not p.exists():
            raise FileNotFoundError(f"会话未开局（缺 observation.json）: {self._id}")
        return Observation.model_validate(json.loads(p.read_text(encoding="utf-8")))

    def _load_engine(self) -> RuleEngine:
        engine = load_rule_engine(str(self._dir / RULES_FILE))
        if engine is None:  # 构造期必落——缺失即损坏（不静默重建：自定义规则不允许被换掉）
            raise RuntimeError(f"rules.pkl 缺失（会话损坏）: {self._id}")
        return engine

    def _write_observation(self, o: Observation) -> None:
        """观测投影落盘（最后写）：prior 冻结牙齿 + recent_decisions 日志尾投影。"""
        p = self._dir / OBSERVATION_FILE
        if p.exists():
            disk = self._load_observation()
            if (disk.prior or None) != (o.prior or None):
                raise ValueError(f"prior 在 episode 内冻结（改卡/改协议重开走新会话）：{self._id}")
        o.recent_decisions = list(_DecisionLog(self._dir / DECISION_LOG_FILE).tail(TOP_K_DECISIONS))
        write_text_atomic(str(p), o.model_dump_json(indent=1))
