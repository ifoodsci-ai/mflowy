"""会话基座机制：POMDPSession（会话壳，#89 + #226 基座重构）+ SessionStore（存量会话 GC 面）。

driver=机制、域=策略——POMDPSession 吃下壳骨架：六个操作上下文承载业务流程
（终态门/预算门/counts_step 单源计步/o_actions 投影/动作面先验冻结/补 r 机制/
append-only 决策日志/统一盘上写序），域实现 ``pomdp.POMDPSessionHandle`` 只填
转移语义。盘上三件（rules.pkl 随规则引擎退役——#120；存量会话的遗留文件不读
不迁，GC 自然回收）：

- ``observation.json``：观测投影（四件，pydantic 封闭模型），整读改写、原子
  落盘——最后写
- ``decision.log``：决策日志（JSONL append-only，一行一条 (s,a,r,s′)——⑥ 补 r
  由壳在 commit 点 stamp）——追加即 commit 点；``ra`` 接口无删除面
- ``memo.json``：盘上 memo（③）——会话路径锚定的观测派生缓存，open 随磁盘 o
  一起水化、写穿原子（模块级内存 dict 的无 gc/同进程跨会话共享两缺陷由此解）

统一写序（构造保证）：工件先行（域、未 commit、孤儿无害）→ 日志追加（commit）→
观测投影最后落盘。崩溃窗口二分：追加前崩 = 失败免费；追加后投影前崩 = 日志领先，
``open`` 的一致性牙齿（validation.recent_decisions 尾 ≡ 日志尾）fail-loud。
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from mflowy.utils.file import write_text_atomic

from .pomdp import TOP_K_DECISIONS, Observation, ObservationValidation

OBSERVATION_FILE = "observation.json"
DECISION_LOG_FILE = "decision.log"
MEMO_FILE = "memo.json"


def _canonical(prior: Mapping[str, Any] | None) -> str | None:
    """先验快照 canonical 形态：sort_keys 排序序列化——嵌套结构内容比对（顺序无关）。"""
    if prior is None:
        return None
    return json.dumps(prior, sort_keys=True, ensure_ascii=False, default=str)


def gc_report_key(dry_run: bool) -> str:
    """gc 报告键随 dry_run 切换：would_delete（只列）/ deleted（已删）——键名单一来源。"""
    return "would_delete" if dry_run else "deleted"


class SessionStore:
    """会话 GC 面（存量兼容）：manifest.json 随 #89 溶解后仅回收语义仍有消费方
    （mcp `sessions_gc`）——生命周期（终态/弃局/IO）已全部住 POMDPSession。

    不读会话内容——损坏/不可读会话照样入删单；mtime 锚 observation.json
    （#89 新格式），回退 manifest.json（存量）、再回退目录自身 mtime。
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def gc(self, ttl_days: int = 30, dry_run: bool = True) -> dict[str, list]:
        """状态无关回收：按龄清理会话目录（dry_run 只列不删——键随 dry_run 切换）；
        root 内非目录条目不动作（``ignored``）；root 不存在 = 空报告。"""
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


class _RewardStampLog:
    """补 r 门面（#226 ⑥）：commit 点把回合奖励 stamp 进记录（键缺省时）——
    (s,a,r,s′) 叙事在日志层可重建（累计连事后重建自此可能）。

    域须先置 ``o.validation.reward`` 再 append（#234 形状适配已改两域调用序）；
    只读面透传内层（审计读与迭代不受包装影响）。
    """

    def __init__(self, inner: _DecisionLog, o: Observation) -> None:
        self._inner = inner
        self._o = o

    def append(self, record: Mapping[str, Any]) -> None:
        if "r" not in record:
            record = {**record, "r": self._o.validation.reward}
        self._inner.append(record)

    def tail(self, n: int) -> list[Mapping[str, Any]]:
        return self._inner.tail(n)

    def __len__(self) -> int:
        return len(self._inner)

    def __iter__(self) -> Iterator[dict]:
        return iter(self._inner)

    def __getitem__(self, i: int) -> dict:
        return self._inner[i]


class SessionMemo(MutableMapping[str, Any]):
    """盘上 memo（#226 ③）：会话路径锚定的观测派生缓存。

    模块级内存 dict（无 gc、同进程跨会话共享——清洗域 _REPORT_MEMO/_PROFILE_MEMO
    旧形态）由此退役的机制载体：open/load_or_start 随磁盘 o 一起水化、写穿原子
    （tmp+rename，红线 3）；observe 以会话路径寻址即读即得。值域 JSON
    （default=str 宽容序列化——与 decision.log 同尺）。
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._flush()

    def __delitem__(self, key: str) -> None:
        del self._data[key]
        self._flush()

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def _flush(self) -> None:
        write_text_atomic(str(self._path), json.dumps(self._data, ensure_ascii=False, indent=1, default=str))


class POMDPSession:
    """POMDPSession 会话壳（#89 统一机制 + #226 基座重构）：所有业务共用，域只填
    转移语义。

    新开 = ``POMDPSession(root, prefix=...)``：mint id + 建目录 + 空决策日志；
    开局观测由 ``start_ctx`` 提交。重水化 = ``POMDPSession.open(observation_json)``：
    入口一致性牙齿（观测投影 ≡ 日志尾）+ memo 水化。生命周期缝 =
    ``load_or_start``（#226 ②/#242——给 observation_json 加载、不给新建）。

    步数口径（#226 ② 单源）：壳按 ``counts_step(action)`` 谓词在 apply 的 commit
    缝计步（域内手动自增退役）；预算门 = counts_step(action) ∧ steps ≥ max_steps。
    o_actions 投影（#226 ④）：``actions_projector`` 域词表钩子（与 counts_step
    谓词同构居所），写点刷新纯词清单（不带吃步标记）。
    """

    def __init__(
        self,
        root: Path | str,
        *,
        prefix: str = "",
        counts_step: Callable[[str], bool] | None = None,
        actions_projector: Callable[[], list[str]] | None = None,
    ) -> None:
        self._root = Path(root).absolute()  # 会话目录绝对路径先归一（②）
        self._counts_step = counts_step
        self._actions_projector = actions_projector
        minted = f"{prefix}{uuid.uuid4().hex[:12]}"
        self._root.mkdir(parents=True, exist_ok=True)
        self._dir = self._root / minted
        self._dir.mkdir(exist_ok=False)  # mint 冲突 fail-loud
        self._id = minted
        (self._dir / DECISION_LOG_FILE).write_text("", encoding="utf-8")
        self._memo = SessionMemo(self._dir / MEMO_FILE)

    @classmethod
    def open(
        cls,
        observation_json: Path | str,
        *,
        counts_step: Callable[[str], bool] | None = None,
        actions_projector: Callable[[], list[str]] | None = None,
    ) -> POMDPSession:
        """重水化既有会话：锚 = 状态文件本体 observation.json 绝对路径（#242——
        bo study_db 同构；目录锚定已退役）；相对路径/名不符/目录/开局观测缺失
        fail-loud；投影与日志撕裂 fail-loud；memo 随磁盘 o 一起水化（③）。"""
        p = Path(observation_json)
        if not p.is_absolute():
            raise ValueError(
                f"observation_json 须为状态文件绝对路径（相对路径在 task_dir 锚定后有歧义）: {observation_json}"
            )
        if p.name != OBSERVATION_FILE or p.is_dir():
            hint = (
                f"（旧目录锚请改传 {observation_json}/{OBSERVATION_FILE}）"
                if p.is_dir()
                else "（会话目录 = 其父目录；目录锚定已退役）"
            )
            raise ValueError(
                f"observation_json 须为 {OBSERVATION_FILE} 文件绝对路径（#242 状态文件本体锚定，"
                f"bo study_db 同构）{hint}: {observation_json}"
            )
        ss = cls.__new__(cls)
        ss._dir = p.parent
        ss._root = ss._dir.parent
        if not ss._dir.is_dir():
            raise FileNotFoundError(f"会话不存在: {ss._dir}")
        if not p.exists():
            raise FileNotFoundError(f"会话未开局（缺 observation.json）: {observation_json}")
        ss._counts_step = counts_step
        ss._actions_projector = actions_projector
        ss._id = ss._dir.name
        ss._memo = SessionMemo(ss._dir / MEMO_FILE)  # 随磁盘 o 一起水化
        obs = ss._load_observation()
        if list(obs.validation.recent_decisions) != _DecisionLog(ss._dir / DECISION_LOG_FILE).tail(TOP_K_DECISIONS):
            raise RuntimeError(f"观测投影与决策日志不一致（撕裂写嫌疑，日志领先）: {observation_json}")
        return ss

    @classmethod
    def load_or_start(
        cls,
        *,
        observation_json: Path | str | None = None,
        root: Path | str | None = None,
        prefix: str = "",
        counts_step: Callable[[str], bool] | None = None,
        actions_projector: Callable[[], list[str]] | None = None,
    ) -> POMDPSession:
        """生命周期缝（#226 ②，Handle load_or_start 的壳面；#242 正名）：
        observation_json 给定 = 加载（状态文件绝对路径、缺失 fail-loud）；
        缺省 = root+prefix 新建。

        加载不吃开局入参——「observation_json + 域尾巴开局入参同传 = 参数来源
        冲突」在域缝 fail-loud（具名先验参数只有域知道）；续局带新认知 = 以旧
        会话产物 data 路径 + 新 prior 开新局（集链语义），不开旧局换卡。
        """
        if observation_json is not None:
            return cls.open(observation_json, counts_step=counts_step, actions_projector=actions_projector)
        if root is None:
            raise ValueError("observation_json 缺省 = 新建会话——须给 root（mint 目录的父）")
        return cls(root, prefix=prefix, counts_step=counts_step, actions_projector=actions_projector)

    @property
    def id(self) -> str:
        """会话目录名（内部寻址/标签/实验名面）。"""
        return self._id

    @property
    def observation_json(self) -> str:
        """Handle 寻址面（#226 ②/#242 正名）：状态文件绝对路径（会话目录/observation.json
        ——bo study_db 同构的状态文件本体锚）。"""
        return str(self._dir / OBSERVATION_FILE)

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def memo(self) -> SessionMemo:
        """盘上 memo（③）：观测派生缓存，写穿原子。"""
        return self._memo

    @property
    def is_terminal(self) -> bool:
        o = self._load_observation()
        return o.validation.finished or o.validation.abandoned

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
    def start_ctx(
        self, max_steps: int, *, prior: Mapping[str, Any] | None = None, biz: Mapping[str, Any] | None = None
    ) -> Iterator[Observation]:
        """开局：body 内物化工件（工件先行），exit 提交观测（开局 commit 点）——
        prior/biz 形状域定（typed Prior 随 #235/#236 实体化）。"""
        if (self._dir / OBSERVATION_FILE).exists():
            raise FileExistsError(f"会话已开局: {self._id}")
        o = Observation(prior=prior, biz=biz or {}, validation=ObservationValidation(max_steps=max_steps))
        yield o
        self._write_observation(o)

    @contextmanager
    def observe_ctx(self) -> Iterator[Observation]:
        """观测（#226 ③ 计算语义）：对观测业务目标计算内存 o 并**更新磁盘 o**。

        「只读」指对真态只读（写的是观测派生面：biz 投影/prior 更新⑧/memo）；
        零转移边界维持——决策日志零追加、不吃步、盘上真态工件不动。终态旗恒
        盘读（body 内清旗不复活终局）；不设终态门——终态会话照常可观测
        （续局/re-orientation 锚点，观测不是动作）。
        """
        o = self._load_observation()
        yield o
        disk = self._load_observation()
        o.validation.finished = disk.validation.finished
        o.validation.abandoned = disk.validation.abandoned
        o.validation.steps = disk.validation.steps  # 不吃步：步数恒盘读（只随 apply commit 缝变化）
        self._write_observation(o)

    @contextmanager
    def apply_ctx(self, action: str) -> Iterator[tuple[Observation, _RewardStampLog]]:
        """执行动作：终态门 + 预算门 + 壳单源计步（②——counts_step 谓词，域内手动
        自增退役）；``l.append`` 写穿即 commit（⑥ 补 r：壳从 o.validation.reward
        stamp 进记录）；exit 投影最后。

        body 内失败即免费（追加前抛出：零日志、零观测变更、孤儿工件无害——计步
        同回滚）；先验在动作面冻结（⑧——init/更新只走 load_or_start/observe 缝）。
        """
        o = self._gate_mutable()
        if self._action_counts(action):
            if o.validation.steps >= o.validation.max_steps:
                raise RuntimeError(f"预算耗尽（max_steps={o.validation.max_steps}）：请 rollback 或 finish")
            o.validation.steps += 1  # 单源记账：commit 缝计步（body 失败 = 盘上未写，免费）
        prior_before = _canonical(o.prior)
        yield o, _RewardStampLog(_DecisionLog(self._dir / DECISION_LOG_FILE), o)
        self._assert_prior_frozen(o, prior_before)
        self._write_observation(o)

    @contextmanager
    def rollback_ctx(self) -> Iterator[tuple[Observation, _RewardStampLog]]:
        """回滚：域在 body 定位目标记录、恢复指针状态、追加 rollback 记录（不删行；
        补 r 同 apply——恢复 ΔΦ 先置再 append）。"""
        o = self._gate_mutable()
        prior_before = _canonical(o.prior)
        yield o, _RewardStampLog(_DecisionLog(self._dir / DECISION_LOG_FILE), o)
        self._assert_prior_frozen(o, prior_before)
        self._write_observation(o)

    @contextmanager
    def finish_ctx(self) -> Iterator[tuple[Observation, _DecisionLog]]:
        """收局：abandoned → raise（改协议重开走新会话）；finished → 只读重入
        （``l.append`` 即 raise、exit 不落盘——幂等重导出由构造保证，交付物
        工件写不受门管）；活会话 → body 置 ``o.validation.finished=True``、追加
        finish 记录，exit 投影（拒收路不置 finished，会话保留可续局）。
        """
        o = self._load_observation()
        if o.validation.abandoned:
            raise RuntimeError(f"会话已弃局：改协议重开走新会话（{self._id}）")
        if o.validation.finished:
            yield o, _DecisionLog(self._dir / DECISION_LOG_FILE, readonly=True)
            return  # 幂等重入：零落盘
        log = _DecisionLog(self._dir / DECISION_LOG_FILE)
        yield o, log
        self._write_observation(o)

    def abandon(self, reason: str | None = None) -> dict[str, Any]:
        """弃局：日志记录 + 观测置 abandoned。幂等（已弃局重复弃 = no-op 回执）；
        已收局不可弃（二终态互斥）。ack 的 observation_json = 状态文件绝对路径
        （Handle 寻址面，#242）。"""
        o = self._load_observation()
        if o.validation.finished:
            raise RuntimeError(f"会话已终态（收局）：不可弃局（{self._id}）")
        if o.validation.abandoned:
            return {
                "observation_json": self.observation_json,
                "abandoned": True,
                "reason": reason,
                "steps": o.validation.steps,
            }
        o.validation.abandoned = True
        _DecisionLog(self._dir / DECISION_LOG_FILE).append({"action": "abandon", "reason": reason})
        self._write_observation(o)
        return {
            "observation_json": self.observation_json,
            "abandoned": True,
            "reason": reason,
            "steps": o.validation.steps,
        }

    # ── 内部件（三个 RWA 原语与门）─────────────────────────────────────────

    def _gate_mutable(self) -> Observation:
        """可变打开的终态门：已终态会话动作面为空（apply/rollback 共用）。"""
        o = self._load_observation()
        if o.validation.finished or o.validation.abandoned:
            raise RuntimeError(f"会话已终态（收局/弃局）：动作面为空（{self._id}）")
        return o

    def _action_counts(self, action: str) -> bool:
        return self._counts_step is None or bool(self._counts_step(action))

    def _assert_prior_frozen(self, o: Observation, prior_before: str | None) -> None:
        """⑧ 动作面先验冻结牙齿（原观测写缝冻结随 observe 更新缝合法化退役）：
        apply/rollback 路径改 prior = 生命周期违规 fail-loud。快照 = canonical
        序列化（嵌套就地改写也可检——浅拷贝比对两侧重叠失效）。"""
        if prior_before != _canonical(o.prior):
            raise ValueError(
                f"prior 在动作面冻结（init/更新走 load_or_start/observe 缝，改卡重开走新会话）：{self._id}"
            )

    def _load_observation(self) -> Observation:
        p = self._dir / OBSERVATION_FILE
        if not p.exists():
            raise FileNotFoundError(f"会话未开局（缺 observation.json）: {self._id}")
        return Observation.model_validate(json.loads(p.read_text(encoding="utf-8")))

    def _write_observation(self, o: Observation) -> None:
        """观测投影落盘（最后写）：recent_decisions 日志尾投影 + o_actions 词表
        投影（④——钩子在座则刷新纯词清单）。"""
        o.validation.recent_decisions = list(_DecisionLog(self._dir / DECISION_LOG_FILE).tail(TOP_K_DECISIONS))
        if self._actions_projector is not None:
            o.actions = [str(a) for a in self._actions_projector()]
        write_text_atomic(str(self._dir / OBSERVATION_FILE), o.model_dump_json(indent=1))
