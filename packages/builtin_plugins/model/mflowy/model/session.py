"""建模会话引擎（model 域，#89 基座上壳 + #226 拆票C #236 实体化）：#62 回路与
状态机——s 元组（光标 + 账本）+ 九动作面。

状态 ``s = (光标 m/H/seed/F, 账本 board, oof, budget, done, history)``；动作面
``A = {select_model, set_hp, set_hp_space, set_seed, select_FE}（配置编辑：零步费
零奖励）∪ {run_hpo, final_train_test}（计算动作：吃步——步数由壳按 counts_step
谓词单源记账，#226 ②）∪ {report_findings}（agent 判读落诊，#226 ④/#236 收编——
零步费）∪ {done}``（= finish 控制面独立入口）。

会话恒**单目标任务**（x_y target 收窄——多目标在会话外三层循环：多目标 →
多模型 → 单目标任务会话，文献综述阶段输出每目标的模型候选）。

光标 **单表 H**（#66 BREAKING）：``H[param]`` 唯一槽位——标量 = pin、dict
（space_to_json 形态）= 空间收窄；set_hp/set_hp_space 同写 H，last-write-wins
显式（decision_log note 记覆盖前后值；#226 ⑥ 决策记录自带 r——壳在 commit 点
stamp，(s,a,r,s′) 可从日志重建）。

盘上（#89 壳三件 + 域状态/工件）：
- observation.json / decision.log / memo.json —— 壳三件（driver.session；rules.pkl
  随规则引擎退役）；prior = 冻结协议（source_file/target/task/card/protocol/
  candidates/id_columns）
- board.json —— 域状态（光标/快照栈/账本/verdict 缓存）；写序 = board 先行 →
  日志追加（commit）→ 观测投影
- split.pkl / study_*.db / loader_*.pkl / plans/ / deliverables/ —— 工件

- **run_hpo**：sqlite study per (family,F) 增量续跑；reward = Δu_inner（新键 =
  u_inner）；账本条目随附证据配置快照（final_train_test 交付面）；findings 浮出
  进 o.biz（π 的过程决策面）
- **final_train_test**（#66 终训走 modeling 工具）：先搜后门店——渲染 per-combo
  计划（cv.load(from_run_id) 按引用 + 证据配置快照 ∪ θ* 全 pin，Conf →
  flow_to_yaml 单源序列化）→ JobProvider DAG 跑终训/终评/副产物 → u/metrics 回读
  → 副产物指针落账 → holdout 终评恰一次 → combo 冻结；reward = 0
- **finish(pdf)**（#66 单动作）：收编 π 产报告进 deliverables + verdict（幂等
  重导出由壳只读重入保证——双 commit 不可能）；素材面走 draw_taylor_diagram /
  modeling_report_skeleton 两工具
- **账本真源** = 盘上 sqlite study（append-only）；rollback 只回退光标——board
  永不与盘分叉（#58 P1 根治）
- **observe**（#117/#120 取代 propose——菜单退役；#226 ③ 计算语义）：对真态
  只读、观测派生面写盘、零转移，读时投影 π 的决策面（本 combo 键 findings——
  #66、账本摘要、has_cat、候选族超参面）；π 自构动作，词表外动作由 apply 边界
  fail-loud 拒绝；收局时机 π 从观测自决
- 划分身份 = cv.split 副产物 split.pkl 的 run_id 引用（开局走节点中间件链条
  物化——mlflow 窗口/log_cv 留痕/print 捕获；终训 plan 以 cv.load 按引用复用）
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from mflowy.data.types import TASKTYPE
from mflowy.driver.pomdp import FinishVerdict, Observation
from mflowy.driver.session import POMDPSession
from mflowy.utils.file import write_bytes_atomic, write_text_atomic

# 重栈链（registry / model._pipeline / model.study / model.types / model.utils）顶层
# import 惰性化（#218）：瘦装环境可 import 本模块、start/observe/select_* 配置动作
# 可用——重栈（sklearn/optuna 随 mlflow/bo 传递、五族随 [local] extra）只在计算
# 动作的真实消费点导入；瘦装+local provider 的计算失败走 extras.LOCAL_INSTALL_HINT。

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # 仅注解面（__future__ annotations 下零运行期 import）
    from mflowy.model.model.types import MetricName

# 九动作面（#226 ④/#236：report_findings 收编——配置面零步费；done = finish 控制面独立入口）；compute = 吃步（counts_step 口径）
ACTIONS = (
    "select_model",
    "set_hp",
    "set_hp_space",
    "set_seed",
    "select_FE",
    "run_hpo",
    "final_train_test",
    "report_findings",
)
COMPUTE_ACTIONS = frozenset({"run_hpo", "final_train_test"})
DEFAULT_N_TRIALS = 10  # run_hpo 缺省搜索档（执行面单源——出题规则侧随菜单退役，#120）
BOARD_FILE = "board.json"


def _root() -> Path:
    """会话 root（#207 锚定链）：``task_anchor()/modeling_sessions``——链序单源住
    task_anchor（无绑定落 ~/.mflowy，缺省路径与旧版逐字相同）。env
    MFLOWY_MODELING_SESSIONS 已退役（BREAKING，见 CHANGELOG）。"""
    from mflowy.utils.path import task_anchor

    return task_anchor() / "modeling_sessions"


def _counts_step(action: str) -> bool:
    """步数口径（壳预算门的域声明）：预算 = 实验数——只有计算动作吃步。"""
    return action in COMPUTE_ACTIONS


def _open(observation_json: str) -> POMDPSession:
    """按 observation_json（状态文件绝对路径，#226 ②/#242 正名）重水化——相对
    路径/名不符由壳 fail-loud。"""
    return POMDPSession.open(observation_json, counts_step=_counts_step)


# ── 域状态文件（board.json）：光标/快照栈/账本/verdict 缓存 ──────────────────


def _board_load(ss: POMDPSession) -> dict:
    return json.loads((ss.dir / BOARD_FILE).read_text(encoding="utf-8"))


def _board_write(ss: POMDPSession, board: dict) -> None:
    write_text_atomic(str(ss.dir / BOARD_FILE), json.dumps(board, ensure_ascii=False, indent=1, default=str))


def _mview(ss: POMDPSession, o: Observation, board: dict) -> dict:
    """manifest 形状的域逻辑视图：prior（冻结协议）+ o.biz 投影 + board 引用。

    board 的 cursor/models/snapshots 以**引用**进视图——域逻辑经 m 的变更即 board
    的变更，commit 时一次落盘。session_id 从壳独立传参穿线（#226 ②——Observation
    不再携带，纯寻址不进观测）。"""
    p, d = o.prior or {}, o.biz
    return {
        "session_id": ss.id,
        **p,
        "split": d.get("split", {}),
        "cursor": board["cursor"],
        "snapshots": board["snapshots"],
        "models": board["models"],
        "verdict": board.get("verdict"),
        "steps": o.validation.steps,
        "max_steps": o.validation.max_steps,
        "source_fingerprint": d.get("source_fingerprint"),  # #236 工件契约锚（开局原始帧一次计算）
        # finished/abandoned 死投影清（#226 ⑥/#236）——引擎事实单源 o.validation
    }


def _state_projection(m: dict) -> dict:
    """o.biz 投影：划分身份 + 光标 + 账本板面 + 指纹（域逻辑读 m）。"""
    return {
        "source_fingerprint": m.get("source_fingerprint"),  # #236 工件契约锚（apply 覆写不丢）
        "split": m["split"],
        "cursor": {k: (dict(v) if isinstance(v, dict) else v) for k, v in m["cursor"].items()},
        "board": _board(m),
        "digest": _digest(m),
    }


def _load_split(m: dict):
    """从会话目录重建 TrainSplit（D 冻结的数据面；spec 含 inner splitter，整体 pickle）。

    X 已无 ID 列（x_y id_columns 剔除在物化期完成——划分/搜索/终训同一数据面，
    无会话内二次剔除）。"""
    with open(_root() / m["session_id"] / "split.pkl", "rb") as f:
        return pickle.load(f)


def _task(m: dict):
    from mflowy.driver.config import parse_enum

    return parse_enum(TASKTYPE, m["task"])


def _metric_of(m: dict) -> MetricName:
    from mflowy.model.model.types import MetricName
    from mflowy.model.model.utils import infer_metric

    return MetricName(m["protocol"]["metric"]) if m["protocol"].get("metric") else infer_metric(_task(m))


def _candidates(m: dict) -> list[str]:
    """候选面（族级）：卡 candidates（prior 结构化清单）∩ 已声明族目录。"""
    from mflowy.model import registry

    fams = registry.families()
    declared = [c["family"] for c in m.get("candidates") or []]
    return sorted(f for f in declared if f in fams)


def _candidate_of(m: dict, family: str) -> dict:
    """结构化候选条目（family 级唯一——baseline/hints/seed 的居住面）。"""
    for c in m.get("candidates") or []:
        if c["family"] == family:
            return c
    raise ValueError(f"候选 `{family}` 不在卡 candidates（可用：{_candidates(m)}）")


def _has_cat(m: dict) -> bool:
    X = _load_split(m).X_train
    return any(not pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)


def _structure_key(family: str, fe: dict) -> str:
    """账本键 = (family, FE 组合)——不同 FE 下成绩并排可比。"""
    return f"{family}|{fe['cat']}|{fe['num']}"


def _key_file(key: str) -> str:
    return key.replace("|", "__")


def _study_db_name(key: str) -> str:
    """study sqlite 文件名（单源）：盘上真源路径、账本 ``study`` 字段、父 run 投影工件名三处同名。"""
    return f"study_{_key_file(key)}.db"


def _family_params(family: str) -> dict:
    """族签名用户超参面——单源委托 registry.family_face（出题规则侧同源现查）。"""
    from mflowy.model import registry

    return registry.family_face(family)


# 空间声明/持久化的单源件住 model/utils：normalize_space / space_to_json /
# space_from_json / MACHINERY_SKIP——session 不留第二份拷贝


# ── 开局：协议谈判在剧本层（model-prior-survey / model-research-review）完成，cv_card 传入即物化冻结 ──


def _parse_split_yaml(cv_card: str) -> dict:
    """cv_card = cv.split 配置的 yaml 明文（如 ``cv: 3\\ntest_size: 0.2``）→ 参数 dict。

    键 ⊆ cv.split 用户参数面（fail-loud 带可用键清单——签名即约束单源）；划分构建
    参数之外的协议（task/metric/candidates/max_steps）走显式参数/card。"""
    import inspect

    import yaml as _yaml
    from mflowy.model.cv.split import split as _split_fn

    if not isinstance(cv_card, str):
        raise ValueError(
            f"cv_card 须为 yaml 明文 str（cv.split 参数映射，如 'cv: 3\\ntest_size: 0.2'），"
            f"got {type(cv_card).__name__}——dict 协议卡形态已退役（协议走显式参数/card）"
        )
    try:
        parsed = _yaml.safe_load(cv_card)
    except _yaml.YAMLError as e:
        raise ValueError(f"cv_card 不是合法 yaml 明文（cv.split 参数映射，如 'cv: 3\\ntest_size: 0.2'）：{e}") from e
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(
            f"cv_card 须为 cv.split 参数的 yaml 映射明文（如 'cv: 3\\ntest_size: 0.2'），got: {type(parsed).__name__}"
        )
    allowed = set(inspect.signature(_split_fn).parameters) - {"x_y"}
    unknown = set(parsed) - allowed
    if unknown:
        raise ValueError(
            f"cv_card 键 {sorted(unknown)} 不在 cv.split 参数面（可用：{sorted(allowed)}）——"
            "划分构建参数之外的协议走 start_session 显式参数/card"
        )
    return dict(parsed)


class ModelingSession:
    """model 域 POMDP 会话实体（#226 ⑨ 合规修复，#236）：typed 六操作、无状态
    observation_json 寻址（#242 正名——状态文件本体锚，bo study_db 同构）。
    先验 = ModelPrior（冻结协议，⑦）——episode 内动作面冻结（牙齿在壳）；
    apply 参数由 agent 基于上下文中的 o 翻译（红线 10——机器面零注入）。"""

    def load_or_start(
        self,
        observation_json: str | None = None,
        *,
        file_path: str | None = None,
        cv_card: str | None = None,
        target: str | None = None,
        task: str | None = None,
        metric: str | None = None,
        candidates: list[dict] | None = None,
        max_steps: int | None = None,
        id_columns: list[str] | None = None,
        goal: str | None = None,
        external_baseline: dict | None = None,
        card: dict | None = None,
    ) -> tuple[str, Observation]:
        """开局/续载（#226 ②⑤，#236 实体化）：observation_json 给定 = 加载（任何
        开局入参同传 = 参数来源冲突 fail-loud；minted-but-unstarted 孤儿目录给专属
        报错——物化中断所致，重开即愈）；缺省 = cv_card 必填新开（D 物化冻结 +
        光标就位，物化收进 start_ctx body——孤儿窗收窄到 mint 级）。
        返回 (observation_json, o)。

        **显式参数 > card（背景）分层缺省**——合并结果冻结进 ModelPrior prior
        （episode 背景状态，⑧动作面冻结牙齿在壳）；两处同显式来源的冲突仍由
        校验门 fail-loud。改协议 = 新 episode（弃局重开）。
        """
        if observation_json is not None:
            conflicts = {
                "file_path": file_path,
                "cv_card": cv_card,
                "target": target,
                "task": task,
                "metric": metric,
                "candidates": candidates,
                "max_steps": max_steps,
                "id_columns": id_columns,
                "goal": goal,
                "external_baseline": external_baseline,
                "card": card,
            }
            given = sorted(k for k, v in conflicts.items() if v is not None)
            if given:
                raise ValueError(
                    f"参数来源冲突：observation_json（加载既有会话）与开局入参 {given} 不可同传——"
                    "续局带新认知 = 以旧会话产物数据路径 + 新协议开新局"
                )
            from pathlib import Path as _Path

            p = _Path(observation_json)
            if p.is_absolute() and p.parent.is_dir() and not p.exists():
                raise ValueError(  # minted-but-unstarted（#236）：确定性判据（目录在、开局观测缺）
                    f"会话目录存在但未开局（minted-but-unstarted——开局物化中断所致），孤儿不可续，请开新局: {p.parent}"
                )
            ss = POMDPSession.load_or_start(observation_json=observation_json, counts_step=_counts_step)
            return ss.observation_json, ss.observation
        goal = goal or ""
        card = card or {}
        if not cv_card or not str(cv_card).strip():
            raise ValueError("cv_card 必填——cv.split 配置 yaml 明文（协议谈判在剧本层 model-prior-survey 完成）")
        if not file_path:
            raise ValueError("file_path 必填——新开局的原始数据源")
        split_params = _parse_split_yaml(cv_card)
        target = target or card.get("target")
        if not target:
            raise ValueError("target 必填——显式参数或 card.target（无静默缺省）")
        task = task if task is not None else card.get("task")
        if not task:
            raise ValueError("task 必填（regression/classification）——显式参数或 card.task")
        # id 剔除面（#112 涟漪）：单源委托 data.excluded_from_features（pk ∪ known_id ∪
        # composite 成员；fk 不剔——join 需求可见且可作类别特征）；显式 id_columns 覆盖卡背景。
        if id_columns is not None:
            id_cols = list(id_columns)
        else:
            from mflowy.data.types import excluded_from_features

            id_cols = excluded_from_features(card)
        df = pd.read_parquet(file_path) if file_path.endswith(".parquet") else pd.read_csv(file_path)

        # 四类相容校验门：task↔dtype/metric↔task/cv↔数据/预算与候选（含 hints ⊆ 签名空间、baseline 恰一）
        from mflowy.model.protocol import CVCard

        # 缺省单源（#83）：未指定即不传——落 CVCard.build 签名缺省；会话层零复述、零 or 合并
        proto_extras: dict = {}
        if "cv" in split_params:
            proto_extras["cv"] = split_params["cv"]
        if max_steps is not None:
            proto_extras["max_steps"] = max_steps
        proto = CVCard.build(
            df,
            target=target,
            task=task,
            metric=metric,
            candidates=candidates,
            goal=goal,
            external_baseline=external_baseline,
            **proto_extras,
        ).as_dict()

        ss = POMDPSession(_root(), prefix="ms_")
        cursor = {"m": None, "H": {}, "seed": None, "F": {"cat": "passthrough", "num": "passthrough"}}
        board = {
            "cursor": cursor,
            # 光标快照：rollback 弹栈只回退光标——账本/study 永不回退（盘上 append-only）
            "snapshots": [{"m": None, "H": {}, "seed": None, "F": dict(cursor["F"])}],
            "models": {},  # 账本：per (family,F) 条目（真源 = 盘上 sqlite study）
            "verdict": None,
        }
        m = {
            "session_id": ss.id,
            "source_file": str(file_path),
            "target": target,
            "card": card,
            "protocol": {**proto, "split_yaml": cv_card, "split_params": split_params},
            "task": None,
            "id_columns": id_cols,
            "split": {},  # 划分身份（run tag 面板）：run_id 引用 + test_size + group_by
            "candidates": list(proto.get("candidates") or []),  # #63 结构化候选（含 baseline/hints/seed）
            "cursor": board["cursor"],
            "snapshots": board["snapshots"],
            "models": board["models"],
            "steps": 0,
            "max_steps": proto["max_steps"],  # 冻结协议值直抄——无二次合并（#83）
        }
        with ss.start_ctx(max_steps=proto["max_steps"]) as o:
            _materialize(ss, m, df, task)  # #236：物化收进 ctx body——孤儿窗收窄（失败留 minted-unstarted，不污染观测）
            o.prior = _build_prior(m, card, id_cols)  # task 在物化后落定（x_y 推断）——prior 随之冻结
            from mflowy.data.fingerprint import fingerprint as _fp

            m["source_fingerprint"] = _fp(df)  # #236 工件契约锚（原始帧一次计算——单源进 m/biz）
            o.biz = _state_projection(m)
            _board_write(ss, board)  # 域状态先行（开局 commit = 观测落盘）
        return ss.observation_json, o

    def apply(self, observation_json: str, action: str, params: dict | None = None) -> Observation:
        """执行一个回合动作：action ∈ 九动作面词表（配置编辑 + 计算动作 + 落诊，
        #226 ④/#236 report_findings 收编；done = finish 控制面独立入口）。

        配置编辑只改光标（零步费零奖励，生效于该键下一次 run_hpo）；计算动作吃步
        （counts_step 口径 → 壳预算门）。词表外 fail-loud 带词表清单（单源 = ACTIONS）。
        """
        params = dict(params or {})
        if action not in ACTIONS:
            raise ValueError(
                f"动作 `{action}` 不在词表（九动作面 = {list(ACTIONS)}；done = finish 控制面独立入口，"
                "rollback/finish/abandon 同为独立入口）"
            )
        ss = _open(observation_json)
        with ss.apply_ctx(action) as (o, log):
            board = _board_load(ss)
            m = _mview(ss, o, board)
            dispatch = {
                "select_model": _select_model,
                "set_hp": _set_hp,
                "set_hp_space": _set_hp_space,
                "set_seed": _set_seed,
                "select_FE": _select_fe,
                "run_hpo": _run_hpo,
                "final_train_test": _final_train_test,
                "report_findings": _do_report_findings,
            }
            dispatch[action](ss, o, m, board, log, params)
        return o

    def rollback(self, observation_json: str) -> Observation:
        """回退光标到上一快照（决策日志追加 rollback 记录；预算不回补；账本/study 永不
        回退——盘上 append-only）。"""
        ss = _open(observation_json)
        with ss.rollback_ctx() as (o, log):
            board = _board_load(ss)
            m = _mview(ss, o, board)
            if len(m["snapshots"]) <= 1:
                raise RuntimeError("无历史版本可回退")
            snap = m["snapshots"].pop()
            m["cursor"] = {
                "m": snap["m"],
                "H": dict(snap["H"]),
                "seed": snap["seed"],
                "F": dict(snap["F"]),
            }
            with ss.commit_lock():  # 键控合并：models/verdict 从盘保留（#225 裁定 3——光标恒内存面）
                disk = _board_load(ss)
                disk["cursor"] = m["cursor"]
                disk["snapshots"] = m["snapshots"]
                m["models"] = disk["models"]
                board["models"] = disk["models"]
                _board_write(ss, disk)
                o.validation.reward = None  # ⑥ 补 r：先置再 append（回滚无过程奖励）
                log.append(
                    _record(
                        log,
                        m,
                        "rollback",
                        {},
                        f"光标弹栈（预算不回补：steps={m['steps']}/{m['max_steps']}；账本与盘上 study 一致）",
                    )
                )
                o.biz = _state_projection(m)
        return o

    def abandon(self, observation_json: str, reason: str | None = None) -> dict[str, Any]:
        """弃局终局：委托壳 abandon（幂等）——置 abandoned、记录进决策日志、保留
        决策日志与模型账本、不导出交付物；已收局不可弃（终态互斥）。"""
        return _open(observation_json).abandon(reason=reason)

    def finish(self, observation_json: str, *, pdf: str) -> FinishVerdict:
        """收局（done，#66 单动作——报告是 π 的产物，引擎只收编不编译）：

        收编 π 产报告进 deliverables（复制 + tmp/rename 原子落盘；文件缺失 fail-loud）
        + verdict（u/findings/rank 一致性）。收编不验内容——报告是给人看的便当，
        数字权威面是 verdict/board/盘上 study。重入 = 幂等重导出——**壳构造保证**
        （finish_ctx 只读重入：双 commit 不可能、观测零落盘；交付报告不可变更——
        重写须弃局重开）。素材面走 draw_taylor_diagram / modeling_report_skeleton
        两工具，finish 只做一次收编。

        无冻结 combo 可交付 → 拒收 verdict（拒收后可续局——先 run_hpo +
        final_train_test；accepted 恒真的例外路径，reason 即续局指引）；有交付面时
        accepted 恒真（holdout 读在 final_train_test 恰一次，finish 只结算）。
        """
        ss = _open(observation_json)
        with ss.finish_ctx() as (o, log):  # 幂等重入 = 壳只读路径（已收局零落盘）
            with ss.commit_lock():  # 收局 = 汇合点：锁内重读并行分支的冻结条目（#225 裁定 3）
                board = _board_load(ss)
                m = _mview(ss, o, board)
                if board.get("verdict"):  # 已结算：幂等重导出（交付报告不可变更——审计面）
                    adopted = board["verdict"]["report"]["source_pdf"]
                    if pdf != adopted:
                        raise ValueError(
                            f"已收局交付报告不可变更（审计面）——重写报告须弃局重开；已收编报告源：{adopted!r}"
                        )
                    return _verdict_response(m, list(ss.decision_log))

                keys = sorted(k for k, v in m["models"].items() if v["frozen"])
                if not keys:
                    return FinishVerdict(  # 拒收：无交付面（先搜后店），拒收后可续局
                        accepted=False,
                        reason="无冻结 combo 可交付——先 run_hpo + final_train_test 再收局",
                        terminal_utility={"steps": m["steps"]},
                        deliverables={},
                        decision_log=list(ss.decision_log),
                    )

                src = Path(pdf)
                if not src.is_file():
                    raise ValueError(
                        f"报告文件不存在：{pdf!r}（先 draw_taylor_diagram + modeling_report_skeleton 拿素材，π 产报告后带路径收局）"
                    )
                report = _adopt_report(ss.dir, src)

                verdict_models = {k: {"u": m["models"][k]["u"], "findings": m["models"][k]["findings"]} for k in keys}
                board["verdict"] = {
                    "deliver_keys": keys,
                    "models": verdict_models,
                    "rank": _rank_consistency(m, keys),
                    "report": report,
                }
                o.validation.finished = True
                _board_write(ss, board)  # 域状态先行（verdict 缓存）
                log.append(_record(log, m, "finish", {"pdf": str(src)}, f"交付 {keys}；报告 {report['file']}"))
                m = _mview(ss, o, board)
                return _verdict_response(m, list(ss.decision_log))

    def observe(self, observation_json: str) -> Observation:
        """观测（幂等只读零转移，#117 取代 propose）：盘上观测 + 读时决策面投影。

        - 本 combo 键 findings（#66 作用域——规则与 LLM 落诊同粒度，证据不跨 FE 组合）
        - 光标 combo_key、账本摘要（family/frozen/u_inner/θ*/n_trials）、has_cat（FE
          cat 槽资格的数据面）
        - face：候选族超参面（动态维 Space 可收窄 / 静态常量可 pin——原菜单的黑盒开箱
          证据上浮；选项面 = 全候选，不截断）
        投影只回调用方（壳 observe_ctx 零写）；终态会话照常可观测。
        """
        ss = _open(observation_json)
        with ss.observe_ctx() as o:
            m = _mview(ss, o, _board_load(ss))
            cursor = m["cursor"]
            key = _structure_key(cursor["m"], cursor["F"]) if cursor["m"] else None
            o.biz = {
                **o.biz,
                "findings": [
                    {"key": k, **f} for k, e in m["models"].items() if k == key for f in (e.get("findings") or [])
                ],
                "statistics": next(
                    ({"key": k, **e["statistics"]} for k, e in m["models"].items() if k == key and e.get("statistics")),
                    None,
                ),  # 机器统计收尾步证据摘要（#226 ⑥ 收拢——判读素材随观测可重读）
                "combo_key": key,
                "models": {
                    k: {
                        "family": e["family"],
                        "frozen": e["frozen"],
                        "u_inner": e.get("u_inner"),
                        "best_params": e.get("best_params") or {},
                        "n_trials_at_last_eval": e.get("n_trials_at_last_eval"),
                    }
                    for k, e in m["models"].items()
                },
                "has_cat": _has_cat(m),
                "face": {family: _space_summary(family) for family in _candidates(m)},
            }
        return o

    def taylor_report(self, observation_json: str, run_ids: list[str]) -> dict:
        """泰勒聚合门面（draw_taylor_diagram 工具面）：ledger 反查 label（combo key）
        + 终训面校验（fail-loud 带可用清单）+ 副产物 fan-in（taylor_byproducts 边函数
        ——y_test 逐位同一断言）+ 图落会话目录。"""
        from mflowy.model.report import taylor_for_runs

        ss = _open(observation_json)
        m = _mview(ss, ss.observation, _board_load(ss))
        rid_to_key = {v["run_id"]: k for k, v in m["models"].items() if v.get("run_id")}
        unknown = [r for r in run_ids if r not in rid_to_key]
        if unknown:
            raise ValueError(
                f"run_ids {unknown} 无终训副产物（不在账本终训面——先 final_train_test）——可用 run：{sorted(rid_to_key)}"
            )
        runs = [(rid_to_key[r], r) for r in run_ids]
        out = taylor_for_runs(runs, m["task"], ss.dir)
        return {**out, "figure": out["figure"] if out["figure"] is None else str(out["figure"])}

    def report_skeleton(self, observation_json: str) -> dict:
        """报告素材样板（#66 只读零副作用）：表格从账本渲染（数字抄写归引擎——π 不
        默写 u_inner/θ*/u/rank）+ 图片清单（各 combo 的 OOF 图集、终训副产物指针、
        泰勒图路径）+ 叙事槽样板（判断归 π）。"""
        ss = _open(observation_json)
        m = _mview(ss, ss.observation, _board_load(ss))
        d = ss.dir
        from mflowy.model.report import TAYLOR_FIGURE  # 图名单源——与写入方共享常量（#83）
        from mflowy.utils.mlflow import download_run_artifact

        taylor_fig = str(d / TAYLOR_FIGURE) if (d / TAYLOR_FIGURE).exists() else None
        # rank：冻结面按 u 降序的名次（1 = 交付最优）；rank_consistency = u_inner/u 秩相关（SH 赌注出账）
        frozen = sorted(
            (k for k, e in m["models"].items() if e.get("frozen") and e.get("u") is not None),
            key=lambda k: m["models"][k]["u"],
            reverse=True,
        )
        rank_of = {k: i + 1 for i, k in enumerate(frozen)}
        board_rows = []
        images = []
        for k, e in sorted(m["models"].items()):
            if e.get("u_inner") is None:
                continue  # 未搜 combo 无观察面
            board_rows.append(
                {
                    "key": k,
                    "family": e["family"],
                    "fe": e["fe"],
                    "u_inner": e["u_inner"],
                    "u": e["u"],
                    "best_params": e["best_params"],  # θ*
                    "std": e["std"],
                    "n_trials": e["n_trials_at_last_eval"],
                    "frozen": e["frozen"],
                    "rank": rank_of.get(k),  # 冻结面名次（未终训 = None——不伪装）
                }
            )
            images.append(
                {
                    "key": k,
                    "run_id": e.get("run_id"),
                    # OOF 图集寻址 = run 工件树（#95 项4）——本地仓库 download 即原路径（零拷贝，只读契约）
                    "oof_plots": [str(download_run_artifact(e["oof"]["run_id"], p)) for p in e["oof"].get("plots", [])],
                    "holdout_byproducts": e.get("byproducts"),
                    "findings": e.get("findings", []),
                }
            )
        return {
            "observation_json": observation_json,
            "tables": {"board": board_rows},
            "rank_consistency": _rank_consistency(m, frozen),
            "images": images,
            "taylor": taylor_fig,
            "narrative_slots": [
                {"slot": "为什么是这个模型", "hint": "对比 board 的 u_inner/u/rank 与泰勒图的误差结构，说明交付选择"},
                {
                    "slot": "局限与风险",
                    "hint": "消费 findings（drift/slice_collapse/error_concentrated）与 rank_consistency",
                },
                {"slot": "下一步建议", "hint": "回流清洗域候选 / 换 FE 组合 / 加深搜索的证据化建议"},
            ],
            "board": _board(m),
            "decision_log": list(ss.decision_log),
        }


def _build_prior(m: dict, card: dict, id_cols: list[str]) -> dict:
    """冻结协议 → ModelPrior（⑦ typed 载体）→ o.prior 通道形态（单源）。"""
    from mflowy.model.prior import ModelPrior

    return ModelPrior(
        source_file=m["source_file"],
        target=m["target"],
        task=m["task"],  # 物化后落定的实际 task（x_y 推断）
        protocol=m["protocol"],
        candidates=m["candidates"],
        card=card,
        id_columns=list(id_cols),
    ).model_dump(mode="json")


def _session_experiment(m: dict) -> str:
    """会话锚定实验（工件根 = 会话目录 .mlruns，M10）：get-or-create，返回 experiment_id。"""
    import mlflow
    from mflowy.utils.mlflow import setup

    setup()
    d = _root() / m["session_id"]
    name = f"modeling_session_{m['session_id']}"
    exp = mlflow.get_experiment_by_name(name)
    return exp.experiment_id if exp else mlflow.create_experiment(name, artifact_location=(d / ".mlruns").as_uri())


def _materialize(ss: POMDPSession, m: dict, df: pd.DataFrame, raw_task: str) -> None:
    """D 物化冻结：x_y + split 走**节点中间件链**（mlflow 窗口 + log_cv 留痕 + print
    捕获）——split.pkl 是 split 步副产物（log_cv 落 run 工件树），会话目录持有同一
    pickle（划分身份 = run_id 引用，终训 plan 的 cv.load 源）。"""
    from mflowy.data.types import TASKTYPE as _TT
    from mflowy.driver.config import NodeConf, parse_enum
    from mflowy.driver.context import Context
    from mflowy.utils.capture import capture_prints
    from mflowy.utils.mlflow import active_experiment_id, set_active_experiment

    raw_task = str(raw_task).strip()
    if raw_task.upper() in _TT.__members__:
        raw_task = raw_task.upper()
    task = parse_enum(_TT, raw_task) if raw_task else None

    # 中间件链的 node run 归会话锚定实验（set_active_experiment + 用毕还原）
    prev_exp = active_experiment_id()
    set_active_experiment(_session_experiment(m))
    try:
        xy_ctx = Context(NodeConf(node_key="x_y", name="session:x_y"))
        split_ctx = Context(
            NodeConf(node_key="cv__split", name="session:split"),
            prevs=[xy_ctx],
            upstream_by_param={"x_y": [xy_ctx]},  # log_cv 物化副产物的 x_y 侧信道（builder 同形播种）
        )
        with capture_prints() as buf:
            xy = xy_ctx.node.run(xy_ctx, load=df, target=m["target"], task=task, id_columns=m["id_columns"])
            xy_ctx.result = xy  # 链直调不自动落 result（Context.run 才落）——log_cv 侧信道依赖显式播种
            # 缺省单源（#83）：物化折数取冻结协议值——不二落 cv.split 签名缺省（两活源即漂移面）
            split_kwargs = {**m["protocol"]["split_params"]}
            split_kwargs.setdefault("cv", m["protocol"]["cv"])
            spec = split_ctx.node.run(split_ctx, x_y=xy, **split_kwargs)
    finally:
        set_active_experiment(prev_exp)
    captured = buf.getvalue().strip()
    if captured:
        logger.info("session_start[%s] 节点链输出：\n%s", m["session_id"], captured)

    m["split"] = {
        "test_size": getattr(spec.outer, "test_size", None),
        "group_by": spec.group_by,
        "run_id": split_ctx.run_id,
    }
    # 血缘认证 post-hoc 补打（节点链窗口外——同 final_train_test 形态）
    import mlflow

    for k, v in _lineage_tags(m, action="session_start", key=None, family=None).items():
        mlflow.MlflowClient().set_tag(m["split"]["run_id"], k, v)
    from mflowy.utils.mlflow import download_run_artifact

    assert split_ctx.run_id is not None  # split 链落 run 后必在（窗内执行）
    with open(download_run_artifact(split_ctx.run_id, "split.pkl"), "rb") as f:
        ts = pickle.load(f)  # 身份按引用：会话 split.pkl ≡ run 副产物（零二次物化）
    write_bytes_atomic(_root() / m["session_id"] / "split.pkl", pickle.dumps(ts, protocol=pickle.HIGHEST_PROTOCOL))
    m["task"] = xy.task.value


def _lineage_tags(m: dict, action: str, key: str | None, family: str | None) -> dict[str, str]:
    """run 血缘认证 tag 面（fluent 窗口与 post-hoc 补打共用单源）。"""
    return {
        "mflowy.session_id": m["session_id"],
        "mflowy.action": action,
        "mflowy.combo_key": key or "",
        "mflowy.family": family or "",
        "mflowy.split_run_id": str(m["split"].get("run_id") or ""),
        "mflowy.split_test_size": str(m["split"].get("test_size")),
        "mflowy.split_group_by": str(m["split"].get("group_by")),
    }


def _session_run(m: dict, run_name: str):
    """会话侧 mlflow 窗口：工件根锚定会话目录（per-session 实验，M10）。"""
    import mlflow

    return mlflow.start_run(experiment_id=_session_experiment(m), run_name=run_name)


# ── apply：九动作面分派（clean 形状；终态/预算门在壳的 apply_ctx） ───────────


def _commit(
    ss, o, m, board, log, action: str, params: dict, note: str = "", reward=None, findings=None, statistics=None
) -> None:
    """统一提交点：锁内键控合并 → board 先行（域状态）→ 日志追加（commit）→
    观测投影更新（ctx exit 落盘）。

    步数由壳按 counts_step 单源记账（#226 ②——COMPUTE_ACTIONS 只进 _counts_step
    谓词，域内自增与 m["steps"] 镜像双份退役）；reward 先置再 append——壳补 r
    机制（#226 ⑥）在 commit 点 stamp 进记录。

    键控合并（#225 裁定 3，#218 落地——并行分支）：commit 进壳提交锁重读盘上
    board——账本条目按 ``models[params["key"]]`` 键控合并（本动作产物入键，他键
    从盘保留——分支间可交换；裁定 3 的交换面 = entry[key]/steps/决策日志）。光标
    恒内存面（last-write-wins——最小装配等运行期游走合法；光标不在交换面内）。
    同键并行不在并行模型内（一局 N 模型设计 = N 键 N 分支；freeze 门按键隔离即
    此约定）。"""
    o.validation.reward = reward
    with ss.commit_lock():
        disk = _board_load(ss)
        disk["cursor"] = m["cursor"]
        disk["snapshots"] = m["snapshots"]
        merged_models = dict(disk["models"])
        if (key := params.get("key")) is not None:
            merged_models[key] = m["models"][key]  # 本动作产物（params.key 约定）
        disk["models"] = merged_models
        board["models"] = merged_models
        m["models"] = merged_models  # 投影随合并面（biz 不漏并行分支条目）
        _board_write(ss, disk)
        log.append(_record(log, m, action, params, note))
        state = _state_projection(m)
        if findings is not None:
            state["findings"] = findings  # 计算动作的过程决策面浮出（π 读数）
        if statistics is not None:
            state["statistics"] = statistics  # 机器统计收尾步证据摘要（判读素材，#226 ⑥ 收拢）
        o.biz = state


def _require_cursor_model(m: dict) -> str:
    fam = m["cursor"]["m"]
    if not fam:
        raise RuntimeError("先 select_model 选定候选——配置编辑与计算动作都作用于当前候选配置")
    return fam


def _push_snapshot(m: dict) -> None:
    c = m["cursor"]
    m["snapshots"].append({"m": c["m"], "H": dict(c["H"]), "seed": c["seed"], "F": dict(c["F"])})


# ── 配置编辑动作（零步费零奖励） ────────────────────────────────────────────


def _select_model(ss, o, m, board, log, params: dict) -> None:
    unknown = set(params) - {"family"}
    if unknown:
        raise ValueError(f"select_model 只收 family，多余参数 {sorted(unknown)}")
    family = params.get("family")
    if not isinstance(family, str) or not family:
        raise ValueError(f"select_model 须带 family（候选：{_candidates(m)}）")
    if family not in _candidates(m):
        raise ValueError(f"候选 `{family}` 未在卡 candidates 声明（开局先验：{_candidates(m)}）")
    _push_snapshot(m)
    m["cursor"]["m"] = family
    if m["snapshots"][-1]["m"] != family:
        # 切族：清 H 后注入该候选的 hints 缺省（卡是先验不是边界——π 回合内可覆盖；
        # 单表 H：标量 hint = pin、空间 hint = 收窄，同槽位直写）
        cand = _candidate_of(m, family)
        m["cursor"]["H"] = dict(cand.get("space_hints") or {})
        if m["cursor"]["seed"] is None and cand.get("seed") is not None:
            m["cursor"]["seed"] = cand["seed"]  # 候选缺省 seed（未显式 set_seed 时生效）
    _commit(ss, o, m, board, log, "select_model", {"family": family}, note=f"光标候选 → {family}", reward=0)


def _slot_repr(v) -> str:
    """光标槽位值的人读形态（覆盖 note 的前后值）：空间 → [lo, hi]/[v1, v2]、标量原样。"""
    if isinstance(v, dict):
        return f"[{v.get('lo')}, {v.get('hi')}]" if v.get("kind") == "continuous" else str(v.get("values"))
    return str(v)


def _set_hp(ss, o, m, board, log, params: dict) -> None:
    if not params:
        raise ValueError("set_hp 须带至少一个 {param: 值} pin")
    family = _require_cursor_model(m)
    from mflowy.model.model.utils import MACHINERY_SKIP

    face = _family_params(family)
    for param in params:
        if param == "random_state":
            raise ValueError("random_state 是种子参数——走 set_seed 专用动作（seed 是搜索的环境参数）")
        if param in MACHINERY_SKIP:
            raise ValueError(f"参数 `{param}` 是 plumbing（{sorted(MACHINERY_SKIP)}）——不可 pin")
        if param not in face:
            raise ValueError(f"not_a_param：`{param}` 不在族 `{family}` 签名超参面（可用：{sorted(face)}）")
    overrides = [
        f"{k}: 空间 {_slot_repr(m['cursor']['H'][k])} → pin {v}"
        for k, v in params.items()
        if isinstance(m["cursor"]["H"].get(k), dict)
    ]
    _push_snapshot(m)
    m["cursor"]["H"].update(params)
    note = f"pin {sorted(params)}"
    if overrides:
        note += f"（显式覆盖：{'; '.join(overrides)}）"
    _commit(ss, o, m, board, log, "set_hp", dict(params), note=note, reward=0)


def _set_hp_space(ss, o, m, board, log, params: dict) -> None:
    if not params:
        raise ValueError("set_hp_space 须带至少一个 {param: [lo, hi] | [v1, v2, …]} 收窄")
    family = _require_cursor_model(m)
    face = _family_params(family)
    normalized = {}
    for param, value in params.items():
        if param not in face:
            raise ValueError(f"not_a_param：`{param}` 不在族 `{family}` 签名超参面（可用：{sorted(face)}）")
        from mflowy.model.model.utils import normalize_space, space_to_json

        normalized[param] = space_to_json(normalize_space(param, face[param], value))
    overrides = [
        f"{k}: pin {m['cursor']['H'][k]} → 空间 {_slot_repr(normalized[k])}"
        for k in normalized
        if m["cursor"]["H"].get(k) is not None and not isinstance(m["cursor"]["H"][k], dict)
    ]
    _push_snapshot(m)
    m["cursor"]["H"].update(normalized)
    note = f"收窄 {sorted(params)}"
    if overrides:
        note += f"（显式覆盖：{'; '.join(overrides)}）"
    _commit(ss, o, m, board, log, "set_hp_space", dict(params), note=note, reward=0)


def _set_seed(ss, o, m, board, log, params: dict) -> None:
    unknown = set(params) - {"seed"}
    if unknown:
        raise ValueError(f"set_seed 只收 seed，多余参数 {sorted(unknown)}")
    seed = params.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed 须为 int（搜索的环境参数——换 seed = 换目标 = study 重开），got {seed!r}")
    _push_snapshot(m)
    m["cursor"]["seed"] = seed
    _commit(
        ss,
        o,
        m,
        board,
        log,
        "set_seed",
        {"seed": seed},
        note=f"seed={seed}（下次 run_hpo 生效；目标变即 study 重开）",
        reward=0,
    )


def _select_fe(ss, o, m, board, log, params: dict) -> None:
    unknown = set(params) - {"node"}
    if unknown:
        raise ValueError(f"select_FE 只收 node（FE 节点名），多余参数 {sorted(unknown)}")
    node = params.get("node")
    from mflowy.model import registry

    vocab = registry.fe_vocab()
    if node not in vocab:
        raise ValueError(f"动作 `{node}` 不在 FE 词表（{sorted(vocab)}）")
    family = _require_cursor_model(m)
    st = registry.families()[family]["mdp_family"]["structure"]
    slot = vocab[node]
    if slot == "cat":
        if st["cat_support"] == "native":
            raise ValueError(f"{family} 原生类别支持（cat_support=native）——cat 编码冗余")
        if not _has_cat(m):
            raise ValueError("数据无类别列——cat 槽无对象")
    elif not st["scaling_need"]:
        raise ValueError(f"{family} 不需缩放（scaling_need=False）——num 槽冗余")
    option = (
        {v: k for k, v in registry.FE_CAT_NODES.items()}[node]
        if slot == "cat"
        else {v: k for k, v in registry.FE_NUM_NODES.items()}[node]
    )
    cur = m["cursor"]["F"][slot]
    if cur == option:
        raise ValueError(f"{slot} 槽已是 {option}——同槽换档 select_FE 另一节点，清槽走 rollback")
    _push_snapshot(m)
    m["cursor"]["F"][slot] = option
    _commit(
        ss,
        o,
        m,
        board,
        log,
        "select_FE",
        {"node": node},
        note=f"{slot} 槽 {cur} → {option}（换 FE 后 run_hpo 出新账本条目）",
        reward=0,
    )


# ── 计算动作：run_hpo（增量搜索 + OOF 观察面） ─────────────────────────────


def _machinery_of(family: str):
    """两行为共享的 estimator 机器——machinery 存在性由族目录准入校验保证（#58 聚合期门）。"""
    from mflowy.model import registry

    meta = registry.families()[family]["mdp_family"]
    return meta["machinery"], meta.get("fit_params")


def _ensure_structure(m: dict, family: str) -> dict:
    """结构相容守卫 + 最小装配（encode 族 × 类别列 × cat 空槽 → run 自带 ordinal）。"""
    from mflowy.model import registry

    fams = registry.families()
    fe = m["cursor"]["F"]
    if registry.valid_structure((fe["cat"], fe["num"], family), _has_cat(m), fams):
        return fe
    st = fams[family]["mdp_family"]["structure"]
    if _has_cat(m) and fe["cat"] == "passthrough" and st["cat_support"] == "encode":
        m["cursor"]["F"]["cat"] = "ordinal"
        return m["cursor"]["F"]
    raise ValueError(
        f"pipeline FE {fe['cat']}/{fe['num']} 与族 `{family}` 结构不相容（相容性 = .mdp_family.structure）"
    )


def _tag_run(run, m: dict, *, action: str, key: str, family: str) -> None:
    """mlflow 血缘认证（fluent 窗口内）：委托 _lineage_tags 单源。"""
    import mlflow

    mlflow.set_tags(_lineage_tags(m, action=action, key=key, family=family))


def _run_hpo(ss, o, m, board, log, params: dict) -> None:
    """run_hpo：**全形态经 JobProvider**（#218 债清偿，用户裁定 2026-11-21——local
    不再进程内直调）：渲染搜索 plan → ``provider.hyperparameters_search`` → 消费
    JobOutcome——finished（进程内 DAG / mock：LocalRunFace 本地库直读）或
    submitted（平台链：阻塞拉日志 → zip 解包 → 查询）。study 采用（``add_trial``
    增量并入会话正典库）+ OOF 工件回读，账本条目两形态同形。"""
    n_trials = params.get("n_trials", DEFAULT_N_TRIALS)
    platform_experiment_id = params.get("platform_experiment_id")
    unknown = set(params) - {"n_trials", "platform_experiment_id"}
    if unknown:
        raise ValueError(
            f"run_hpo 只收 n_trials / platform_experiment_id（remote 提交面；pin/收窄走 set_hp/set_hp_space），"
            f"多余参数 {sorted(unknown)}"
        )
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError(f"n_trials 须为 ≥1 的 int，got {n_trials!r}")

    family = _require_cursor_model(m)
    if family not in _candidates(m):
        raise ValueError(f"候选 `{family}` 未在卡 candidates 声明（开局先验：{_candidates(m)}）")
    fe = _ensure_structure(m, family)
    key = _structure_key(family, fe)
    entry = m["models"].setdefault(key, _new_entry(key, family, fe))
    if entry["frozen"]:
        raise RuntimeError(f"combo `{key}` 已冻结（final_train_test 后同键不可再搜索——掐断用 test 反复挑参）")

    from .job_provider import get_job_provider
    from .job_provider.local import LocalJobProvider

    provider = get_job_provider()
    if isinstance(provider, LocalJobProvider):
        # 瘦装 + local 永久降级路径（#217 ④/#218 ④）：重栈 import 预检——DAG 内失败
        # 只余聚合 repr，提交前给安装提示（model_cls 工厂急切导入，无 fit）
        machinery, _ = _machinery_of(family)
        try:
            machinery(_task(m), None)
        except ModuleNotFoundError as e:
            from .extras import LOCAL_INSTALL_HINT

            raise RuntimeError(f"{LOCAL_INSTALL_HINT}（缺 {e.name}）") from e

    _run_hpo_provider(
        ss,
        o,
        m,
        board,
        log,
        entry=entry,
        family=family,
        fe=fe,
        key=key,
        n_trials=n_trials,
        platform_experiment_id=platform_experiment_id,
    )


def _run_hpo_provider(ss, o, m, board, log, *, entry, family, fe, key, n_trials, platform_experiment_id) -> None:
    """run_hpo 的 provider 执行面（#218/#225 裁定 1）：plan 渲染（remote_transport
    provider → uploads 寻址 + cv.split 确定性复现；其余 → 本径 + cv.load 按引用
    复用会话划分）→ 提交 → 消费 → study 采用 + OOF 回读 → 账本更新。

    全 pin 退化形（有效空间空 × n_trials=1）：单值 DiscreteSpace 渲染保持 search
    行为（DAG 两行为判据下裸全 pin 会转 training——碰 holdout，越权）；无动态维
    的族不可退化评估（fail-loud）。稳态链 = ``inherit_platform_job_id``（工作区
    恢复）+ plan 内 ``inherit_study_run_id``（study 寻址，#244 外部锚）两层。"""
    from mflowy.model.model.study import STUDY_NAME, ContinuousSpace, DiscreteSpace
    from mflowy.model.model.types import MetricName

    from . import session_remote
    from .job_provider import get_job_provider
    from .job_provider.remote import RemoteJobProvider

    scoring = _metric_of(m)
    seed = m["cursor"]["seed"]
    objective_id = {"seed": seed if seed is not None else "default", "scoring": scoring.value}
    chain = entry.get("chain") or {}
    n_adopted = chain.get("n_adopted", 0)

    # 链外历史门（红线 1 宁可拒绝）：盘上正典 study 有链簿外 trial（provider 化之前
    # 的存量会话 / 手工改动）——拒绝（续跑要求该键历史全在 provider 链上）
    session_db = _root() / m["session_id"] / _study_db_name(key)
    if session_db.exists():
        import optuna

        names = optuna.get_all_study_names(storage=f"sqlite:///{session_db}")
        if key in names:
            opened = optuna.load_study(study_name=key, storage=f"sqlite:///{session_db}")
            n_local = len(opened.trials)
            # 已采用基线 = 链簿 ∪ study 侧留痕（自愈面：采用后、commit 前崩溃按 study 真值放行重试）
            adopted_baseline = max(n_adopted, int(opened.user_attrs.get("mflowy.remote_adopted", 0) or 0))
        else:
            n_local = adopted_baseline = 0
        if n_local != adopted_baseline:
            raise RuntimeError(
                f"combo `{key}` 盘上 study 有链外历史（trials={n_local}，链内已采用={adopted_baseline}）——"
                "provider 化之前的存量会话不可续跑：换 FE/seed 出新键，或弃局重开"
            )

    # 有效空间 / 退化形判定（单源：face 动态维 ∪ 光标空间槽位）
    face = _family_params(family)
    pinned = {k for k, v in m["cursor"]["H"].items() if not isinstance(v, dict)}
    dynamic_face = {k for k, v in face.items() if isinstance(v, (ContinuousSpace, DiscreteSpace))}
    effective_space = {k for k in dynamic_face if k not in pinned} | {
        k for k, v in m["cursor"]["H"].items() if isinstance(v, dict)
    }
    degenerate_dim = None
    if not effective_space:
        if n_trials != 1:
            raise ValueError("全 pin 候选（有效空间为空）走 n_trials=1 退化形——多余 trial 是同一参数的重复评估")
        from mflowy.model import registry

        # 单点渲染面：优先 pinned 动态维；静态缺省族（如 MLP）退到签名类型面收
        # Space 的维（registry.family_space_dims——值取 pin 或声明缺省）
        degenerate_dim = next(iter(sorted(pinned & dynamic_face)), None) or next(
            iter(sorted(registry.family_space_dims(family))), None
        )
        if degenerate_dim is None:
            raise ValueError(
                f"族 `{family}` 签名无 Space 类型维——全 pin 评估无空间可渲染（search 行为不可达），不可 run_hpo"
            )

    provider = get_job_provider()
    remote_transport = getattr(provider, "remote_transport", False)  # provider 声明面（自定义远程 provider 可标）
    if remote_transport and not platform_experiment_id:
        raise ValueError(
            "remote provider 下 run_hpo 须带 platform_experiment_id——先 ensure_platform_experiment"
            f"（files 含源数据 {Path(m['source_file']).name}）拿到 id 后随 run_hpo 传入"
        )

    plan_text = _hpo_plan_yaml(
        m,
        fe,
        family=family,
        n_trials=n_trials,
        seed=seed,
        scoring=scoring,
        inherit_study_run_id=chain.get("run_id"),
        remote_transport=remote_transport,
        degenerate_dim=degenerate_dim,
    )
    outcome = _run_bridge(
        provider.hyperparameters_search(
            modeling_steps=plan_text,
            name=f"run_hpo:{m['session_id'][:6]}",
            desc=f"会话 {m['session_id']} combo {key} 搜索（provider 形态，#218）",
            platform_experiment_id=platform_experiment_id,
            inherit_platform_job_id=chain.get("job_id"),
        )
    )

    if outcome.status == "finished":
        dag = outcome.dag_result
        if dag is None or dag.status != "finished":
            raise RuntimeError(
                f"搜索受理 DAG 未完成（status={getattr(dag, 'status', None)}）：{getattr(dag, 'error', '')}"
            )
        node = next((r for r in dag.runs if r.node == f"model__{family}" and r.status == "FINISHED" and r.run_id), None)
        if node is None:
            raise RuntimeError(
                f"受理 DAG 无完成的 model__{family} 节点（runs：{[(r.node, r.status) for r in dag.runs]}）"
            )
        face_run = session_remote.LocalRunFace(run_id=node.run_id)
        job_id = None
    else:
        assert outcome.job_id and platform_experiment_id  # submitted 形态构造保证（JobOutcome 互斥不变量）
        client = provider.client if isinstance(provider, RemoteJobProvider) else session_remote.platform_client()
        session_remote.await_job(client, outcome.job_id)
        unpacked = session_remote.fetch_workspace(client, outcome.job_id, platform_experiment_id, ss.dir / "jobs")
        face_run = session_remote.find_model_run(unpacked, family)
        job_id = outcome.job_id

    # 产品回读：study db（链锚定位——#244 外部锚续跑不落本 run 根；树主 = 链簿锚定面）
    study_db_src, study_owner = session_remote.study_db_of(face_run, chain.get("run_id"))
    face_run.artifact("oof_y_pred.parquet")  # 在场过门
    half_width = session_remote.oof_half_width_of(face_run)

    # study 采用（provider→会话正典库翻回连续性）+ 增量并入
    direction = "maximize" if MetricName.higher_is_better(scoring) else "minimize"
    study, n_new = session_remote.adopt_study_trials(
        session_db,
        study_name=key,
        objective_id=objective_id,
        direction=direction,
        seed=seed,
        remote_db=study_db_src,
        n_adopted=n_adopted,
        remote_study_name=STUDY_NAME,
    )

    if outcome.status == "finished":
        # 本地库 run——oof.run_id 直指（零镜像）；后补血缘/指纹/正典投影（#95 项5 同语义）
        import mlflow

        client = mlflow.MlflowClient()
        for k, v in _lineage_tags(m, action="run_hpo", key=key, family=family).items():
            client.set_tag(face_run.run_id, k, v)
        client.set_tag(face_run.run_id, "mflowy.source_fingerprint", m["source_fingerprint"])
        client.log_artifact(face_run.run_id, str(session_db))
        oof_run_id = face_run.run_id
    else:
        oof_src_path = face_run.artifact("oof_y_pred.parquet")
        assert oof_src_path is not None  # 上行在场过门已抛（strict 缺省）——收窄
        oof_run_id = _mirror_oof_run(m, family, key, oof_src=oof_src_path, half_width=half_width, face=face_run)

    # 机器统计收尾步（裁定 2026-11-22）：宿主 = OOF 证据 run——图集/证据表落其工件树
    stats = {"key": key, **_statistics_tail(m, oof_run_id, "oof_y_pred.parquet")}

    from mflowy.model.model._pipeline import u_inner_of

    prev_u = entry.get("u_inner")
    new_u = u_inner_of(study, scoring)
    best = study.best_trial
    assert best.value is not None  # add_trial 拷贝保持完成态（value 非空）
    scores = best.user_attrs.get("scores", [])
    import numpy as np

    # 证据快照：签名静态缺省 ∪ 光标标量 pin（face 键内——与 machinery 时代等价）
    evidence = {k: v for k, v in face.items() if not isinstance(v, (ContinuousSpace, DiscreteSpace))}
    evidence.update({k: v for k, v in m["cursor"]["H"].items() if not isinstance(v, dict)})
    entry.update(
        {
            "study": _study_db_name(key),
            "objective": objective_id,
            "n_trials_at_last_eval": len(study.trials),
            "u_inner": new_u,
            "best_params": dict(best.params),
            "evidence": evidence,
            "std": round(float(np.std(scores)), 4) if len(scores) else None,
            "oof": {
                "run_id": oof_run_id,
                "y_pred": "oof_y_pred.parquet",
                "y_proba": "oof_y_proba.parquet"
                if face_run.artifact("oof_y_proba.parquet", strict=False)
                else None,  # 分类独有（对齐裁定：伴生工件存在性）
                "half_width": half_width,
                "plots": stats["plots"],  # 机器统计收尾步产的图集（落 oof run 工件树，#226 ⑥ 收拢）
            },
            "statistics": stats,  # 证据摘要（evidence_head/n/run_id——observe 投影、判读素材）
            "findings": [f for f in entry.get("findings", []) if f.get("source") == "llm"],
            # provider 链簿（下次提交的 inherit 源——run_id 恒锚 study 树主：续跳
            # 外部锚不换树，本 run 无自有 db 不可寻址）
            "chain": {
                "job_id": job_id,
                "run_id": study_owner,
                "platform_experiment_id": platform_experiment_id,
                "n_adopted": n_new,
            },
        }
    )
    reward = new_u if prev_u is None else round(new_u - prev_u, 4)
    _commit(
        ss,
        o,
        m,
        board,
        log,
        "run_hpo",
        {"n_trials": n_trials, "key": key, "platform_experiment_id": platform_experiment_id},
        note=f"u_inner={new_u}（Δ={reward}）trials={len(study.trials)} run={face_run.run_id[:8]}",
        reward=reward,
        statistics=stats,
    )


def _statistics_tail(m: dict, run_id: str, artifact: str) -> dict:
    """计算动作的机器统计收尾步（#226 ⑥ 承接面收拢，用户裁定 2026-11-22：机器面
    统计 = 动作最后一步，不是独立动作）：在宿主 run（oof/终训）窗口内直调
    ``model__statistics`` 节点——证据表/图集落**同一 run** 工件树（零拷贝、寻址面
    不变），证据摘要随响应/观测浮出（biz.statistics），判读仍归 agent 经
    report_findings。消费面分析（工件已本地、pandas/图集量级）——进程内执行，
    非受理提交。指纹门随节点（宿主 tag 由动作尾先行旁记）。"""
    from mflowy.driver.config import node_key
    from mflowy.driver.discover import get_node

    src_df = (
        pd.read_parquet(m["source_file"]) if m["source_file"].endswith(".parquet") else pd.read_csv(m["source_file"])
    )
    fn = get_node(node_key("model", "statistics")).fn
    import mlflow

    # resume 跨实验 run 前对齐 fluent 活动实验（ensure_experiment 的 set_experiment
    # 全局残留会撞「active experiment does not match」门——DAG 实验残留 vs 会话实验）
    mlflow.set_experiment(experiment_id=mlflow.get_run(run_id).info.experiment_id)
    with mlflow.start_run(run_id=run_id):
        return dict(
            fn(
                src_df,
                run_id=run_id,
                y_pred_artifact=artifact,
                target=m["target"],
                task=m["task"],
                top_k=5,
            )
        )


def _mirror_oof_run(m: dict, family: str, key: str, *, oof_src: Path, half_width, face) -> str:
    """remote 分支的本地镜像 run（输出对齐 local，#225 裁定 1）：lineage tags +
    源指纹 + OOF 证据 parquet + 半宽 tag + 会话 study db 只读投影——账本 oof.run_id
    指向本地库，下游（统计工具/报告/终训复制面）零分叉。返回镜像 run_id。"""
    import mlflow
    from mflowy.model.model._pipeline import OOF_HALF_WIDTH_TAG
    from mflowy.utils.mlflow import log_file

    with _session_run(m, f"run_hpo:{key}") as run:
        _tag_run(run, m, action="run_hpo", key=key, family=family)
        log_file(oof_src, artifact_file="oof_y_pred.parquet")
        if (oof_proba_src := face.artifact("oof_y_proba.parquet", strict=False)) is not None:
            log_file(oof_proba_src, artifact_file="oof_y_proba.parquet")  # 伴生概率面随镜像（存在性——分类独有）
        if half_width is not None:
            mlflow.set_tag(OOF_HALF_WIDTH_TAG, json.dumps(half_width))
        log_file(_root() / m["session_id"] / _study_db_name(key))
        mlflow.set_tag("mflowy.source_fingerprint", m["source_fingerprint"])
    return run.info.run_id


def _run_bridge(coro):
    """同步语境调用 async JobProvider 的 asyncio 桥：无运行循环直接 run；
    已在事件循环内（异步语境直调）借独立线程跑自己的循环。"""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def _final_train_test(ss, o, m, board, log, params: dict) -> None:
    """final_train_test：先搜后门店——终训走 modeling 工具（JobProvider DAG）。

    #66 终训即 plan 执行：渲染 per-combo 计划（划分 cv.load(from_run_id) 按引用
    复用 + 证据配置全 pin，Conf → flow_to_yaml 单源序列化）→ DAG 跑终训/终评/
    副产物/artifacts 一体 → u/metrics 从 DAGResult 回读（ModelLoader.holdout）
    → 副产物指针经 run artifacts 清单解析落账（s 自描述）→ combo 冻结。
    交付的 plan_yaml 就是那次终训本身。

    受理单源 provider（#218 门拆除，#156「终训恒 local」翻转——#225 裁定 5）：
    finished → 进程内 DAGResult 直消费；submitted → remote 分支同机械（拉日志→
    zip→解包→查询→holdout 指标/模型/副产物镜像本地 run），账本条目两分支同形。
    """
    platform_experiment_id = params.get("platform_experiment_id")
    unknown = set(params) - {"platform_experiment_id"}
    if unknown:
        raise ValueError(
            f"final_train_test 只收 platform_experiment_id（remote 提交面；θ* 恒从账本读——先搜后店），"
            f"多余参数 {sorted(unknown)}"
        )
    family = _require_cursor_model(m)
    fe = m["cursor"]["F"]
    key = _structure_key(family, fe)
    entry = m["models"].get(key)
    searched = sorted(k for k, v in m["models"].items() if v.get("u_inner") is not None and not v["frozen"])
    if entry is None or entry.get("u_inner") is None:
        raise ValueError(
            f"先搜后店：combo `{key}` 账本无 cv_score（固定超参候选走 run_hpo n_trials=1 退化形；已搜可店：{searched}）"
        )
    if entry["frozen"]:
        raise RuntimeError(f"combo `{key}` 已冻结（holdout 终评恰一次——test 读即终局）")

    # 证据配置 = 搜索时刻快照 ∪ θ*（搜后改 pin 不进唯一一次 holdout 读）；
    # seed 归一 random_state（搜索环境种子是 θ* 的身份分量，plan 显式携带）
    final_params = {**(entry.get("evidence") or {}), **(entry["best_params"] or {})}
    seed = entry.get("objective", {}).get("seed")
    if isinstance(seed, int):
        final_params["random_state"] = seed

    loader_v = 1  # 冻结世代：freeze-once ⇒ 恒 1（若未来解冻重训，世代递增——指针永不悬空的结构位）
    from .job_provider import get_job_provider

    provider = get_job_provider()
    if getattr(provider, "remote_transport", False):
        # remote 形态 plan：uploads 寻址 + cv.split 确定性复现（无本地 run 引用可依）
        from mflowy.utils.constants import REMOTE_UPLOADS_DIR

        split_params = {**m["protocol"]["split_params"]}
        split_params.setdefault("cv", m["protocol"]["cv"])
        plan_flow = _flow_prefix(
            m,
            fe,
            file_path=f"{REMOTE_UPLOADS_DIR}/{Path(m['source_file']).name}",
            split_params=split_params,
        )
        plan_text = _final_flow_yaml(plan_flow, family, key, final_params)
    else:
        plan_text = _plan_yaml(m, key, final_params)
    plan_path = ss.dir / "plans" / f"final_{_key_file(key)}_v{loader_v}.yaml"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(plan_path, plan_text)  # 审计工件（执行合同面是明文文本，M4 计划落盘留痕）

    result = _run_bridge(
        provider.modeling(
            modeling_steps=plan_text,
            name=f"final_train_test:{m['session_id'][:6]}",
            desc=f"会话 {m['session_id']} combo {key} 终训（终训即 plan 执行）",
            platform_experiment_id=platform_experiment_id,
            inherit_platform_job_id=(entry.get("chain") or {}).get("job_id"),  # 分支 lineage 链
        )
    )
    if result.status == "finished" and result.dag_result is not None:
        dag = result.dag_result
        if dag.status != "finished":
            raise RuntimeError(f"终训 DAG 未完成（status={dag.status}）：{dag.error}")
        nodes = [r for r in dag.runs if r.node == f"model__{family}"]
        node = next((r for r in nodes if r.status == "FINISHED" and r.run_id), None)
        if node is None:
            raise RuntimeError(f"终训 DAG 无完成的 model__{family} 节点（runs：{[(r.node, r.status) for r in nodes]}）")
        loader = node.value
        holdout = getattr(loader, "holdout", None)
        if not holdout:
            raise RuntimeError("终训产物缺 holdout 快照（ModelLoader.holdout）——provider 实现与契约不符，请反馈")
        target_run = node.run_id
    elif result.status == "submitted":
        target_run, loader, holdout = _final_train_test_remote(
            ss, m, family, key, outcome=result, scoring=_metric_of(m)
        )
    else:
        raise RuntimeError(f"终训受理面意外形态（status={result.status}）——JobOutcome 契约违规")

    # 副产物指针经 run artifacts 清单解析落账（s 自描述——board 可见 run 下有什么）
    from mflowy.model.model._pipeline import HOLDOUT_BYPRODUCTS, OOF_HALF_WIDTH_TAG, holdout_byproducts
    from mflowy.utils.mlflow import download_run_artifact, run_artifacts

    byproducts = holdout_byproducts(_task(m))  # 任务感知（对齐裁定：分类加 y_proba 副产物）
    arts = run_artifacts(target_run)
    missing = set(byproducts.values()) - set(arts)
    if missing:
        raise RuntimeError(f"终训 run 缺副产物 {sorted(missing)}（run artifacts：{arts}）——training 行为契约不符")
    pd.read_parquet(
        download_run_artifact(target_run, HOLDOUT_BYPRODUCTS["y_pred"])
    )  # 副产物可读性过门（#236：判读退役后不再消费值）

    # 血缘认证 post-hoc 补打（DAG run 窗口外）+ #164 残差源寻址（复制面）：
    # OOF 证据 parquet + 半宽 tag 随终训 run 复制——predict 单 run_id 入参可用
    # （loader 读终训 run 即得区间原料，final 不再回指 run_hpo run）；remote 分支
    # 的镜像 run 同面收口（本地库 run——同机械零分叉）
    import mlflow

    client = mlflow.MlflowClient()
    for k, v in _lineage_tags(m, action="final_train_test", key=key, family=family).items():
        client.set_tag(target_run, k, v)
    # #236 工件契约：终训 run 旁记原始帧指纹（锚单源 = m.source_fingerprint，与 oof run 同尺）
    client.set_tag(target_run, "mflowy.source_fingerprint", m["source_fingerprint"])
    oof = entry.get("oof") or {}
    if (oof_name := oof.get("y_pred")) and (oof_src := oof.get("run_id")):
        src_arts = run_artifacts(oof_src)
        if oof_name not in src_arts:
            raise RuntimeError(
                f"OOF 证据工件 {oof_name} 不在 run_hpo run {oof_src} 工件树（{src_arts}）——账本与 run 面不一致，请反馈"
            )
        client.log_artifact(target_run, download_run_artifact(oof_src, oof_name))
    if (oof_proba_name := oof.get("y_proba")) and oof_src:
        client.log_artifact(target_run, download_run_artifact(oof_src, oof_proba_name))  # 概率面随终训复制（存在性）
    if (w := oof.get("half_width")) is not None:
        client.set_tag(target_run, OOF_HALF_WIDTH_TAG, json.dumps(w))

    # 机器统计收尾步（裁定 2026-11-22，对称 run_hpo）：宿主 = 终训 run（holdout 面
    # ——y_pred 副产物寻址，指纹 tag 已旁记）；图集/证据表落终训 run 工件树
    stats = {"key": key, **_statistics_tail(m, target_run, byproducts["y_pred"])}
    save_loader(m["session_id"], loader, key, v=loader_v)
    entry.update(
        {
            "frozen": True,
            "u": holdout["u"],
            "metrics": {t: {k.value: v for k, v in tm.items()} for t, tm in holdout["metrics"].items()},
            "findings": [f for f in entry.get("findings", []) if f.get("source") == "llm"],  # 单源 llm（#226 ⑥）
            "run_id": target_run,
            "final_params": final_params,  # 证据配置（= plan 全 pin 面——重放承诺与执行事实同一）
            "byproducts": byproducts,
            "loader_v": loader_v,  # 冻结世代（loader/plan 文件名版本源）
            "statistics": stats,  # holdout 证据摘要（判读素材——报告叙事的误差结构源）
        }
    )
    _commit(
        ss,
        o,
        m,
        board,
        log,
        "final_train_test",
        {"key": key},
        note=f"u={holdout['u']} combo 冻结 run={target_run[:8]}（终训即 plan：{plan_path.name}）",
        reward=0,
        statistics=stats,
    )


def _new_entry(key: str, family: str, fe: dict) -> dict:
    """账本条目初值（真源 = 盘上 study；u_inner/θ* 为派生缓存）。"""
    return {
        "key": key,
        "family": family,
        "fe": dict(fe),
        "study": None,
        "objective": None,
        "n_trials_at_last_eval": 0,
        "u_inner": None,
        "best_params": None,
        "evidence": None,
        "std": None,
        "frozen": False,
        "u": None,
        "metrics": None,
        "findings": [],
        "run_id": None,
        "final_params": None,
        "byproducts": None,
        "oof": None,
        "chain": None,  # provider 链簿（job_id/run_id/n_adopted——下次提交 inherit 源）
    }


# ── 控制面 ──


def _do_report_findings(ss, o, m, board, log, params: dict) -> None:
    """report_findings 动作（#226 ④/#236 收编为第九词——工具面薄别名透传到此）：
    agent 判读落诊（#36 控制面；#226 ⑥ 判读退役后**单源** source=llm——规则/LLM
    双源聚合消失，机器统计证据随计算动作收尾步产出（biz.statistics/observe）后由 agent 判读）。

    schema = diagnose 族 ErrorReport 先例（kind/severity/evidence/message）+
    ``source`` 引擎保留键；kind ⊆ 已知词表（taxonomy fail-loud 不变——语言超纲即拒）。
    """
    key = params.get("key")
    findings = params.get("findings") or []
    if not key:
        raise ValueError("report_findings 须带 key（落诊账本键）")
    if key not in m["models"]:
        raise ValueError(f"落诊键 `{key}` 不在账本（已训 combo：{sorted(m['models'])}）")
    validated = [_validate_llm_finding(f) for f in findings]

    entry = m["models"][key]
    entry["findings"] = validated  # 单源 source=llm（#226 ⑥——双源聚合退役）
    _commit(
        ss,
        o,
        m,
        board,
        log,
        "report_findings",
        {"key": key, "n": len(validated)},
        note="; ".join(f["kind"] for f in validated),
        findings=validated,
    )


def _validate_llm_finding(f: dict) -> dict:
    """schema 校验（fail-loud 带可用面）：kind ⊆ 已知词表；severity ∈ [0,1]；未知顶层键拒绝。"""
    from mflowy.model.taxonomy import KNOWN_FINDING_KINDS

    if not isinstance(f, dict):
        raise TypeError(f"finding 须为 Mapping（ErrorReport 先例 schema），got {type(f).__name__}")
    allowed = {"kind", "severity", "evidence", "message"}
    unknown = set(f) - allowed
    if unknown:
        raise ValueError(f"finding 未知键 {sorted(unknown)}（可用：{sorted(allowed)} + 引擎 source 键）")
    kind = f.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"finding.kind 必填且为非空 str，got {kind!r}")
    if kind not in KNOWN_FINDING_KINDS:
        raise ValueError(f"finding.kind `{kind}` 超纲（#35 定案词表：{sorted(KNOWN_FINDING_KINDS)}）")
    sev = f.get("severity", 0.0)
    if isinstance(sev, bool) or not isinstance(sev, (int, float)) or not 0 <= sev <= 1:
        raise ValueError(f"finding.severity 须为 [0, 1] 数值（缺省 0），got {sev!r}")
    ev = f.get("evidence")
    if ev is not None and not isinstance(ev, dict):
        raise TypeError(f"finding.evidence 须为 Mapping | None，got {type(ev).__name__}")
    msg = f.get("message")
    if msg is not None and not isinstance(msg, str):
        raise TypeError(f"finding.message 须为 str | None，got {type(msg).__name__}")
    return {"source": "llm", **f}


# ── finish（done）：一次结算、多模型交付 ────────────────────────────────────


def _record(log, m: dict, action: str, params: dict, note: str) -> dict:
    """决策记录单源形状（_commit 与 rollback/finish 控制面共用）。"""
    return {
        "step": len(log) + 1,
        "action": action,
        "params": params,
        "steps": m["steps"],
        "budget_left": m["max_steps"] - m["steps"],
        "note": note,
    }


def _adopt_report(session_dir: Path, src: Path) -> dict:
    """收编 π 产报告：复制进 deliverables/（tmp + rename 原子落盘——引擎写走引擎纪律）。

    保留原文件名（π 命名有语义）；tmp 同目录保证 rename 原子（读者只见完整交付物）。
    """
    import shutil

    out_dir = session_dir / "deliverables"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / src.name
    tmp = out_dir / f".{src.name}.tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)
    return {"file": f"deliverables/{src.name}", "source_pdf": str(src)}


def _rank_consistency(m: dict, keys: list[str]) -> float | None:
    """SH 赌注出账：交付集上 u_inner 与 u 的秩相关。样本 < 2 返回 None——不伪装。"""
    import scipy.stats as st

    pairs = [(m["models"][k]["u_inner"], m["models"][k]["u"]) for k in keys]
    if len(pairs) < 2:
        return None
    rho = st.spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic  # type: ignore[reportAttributeAccessIssue]  # scipy 无 py.typed：.statistic 运行期存在
    return round(float(rho), 4) if rho == rho else None


def _verdict_response(m: dict, decision_log: list) -> FinishVerdict:
    """FinishVerdict（从 board 的 verdict 缓存确定性重建——幂等重导出的来源）。"""
    v = m["verdict"]
    keys = v["deliver_keys"]
    deliverables = {
        k: {
            "plan_yaml": _plan_yaml(m, k),
            # loader 本体绝对路径（#242 债清偿：session_id/目录名两段式相对引用退役——
            # 机器铸造原样可持；文件名自带冻结世代指针（#89 M5））
            "loader": str(
                _root() / m["session_id"] / f"loader_{_key_file(k)}_v{m['models'][k].get('loader_v', 1)}.pkl"
            ),
            "run_id": m["models"][k]["run_id"],
            "terminal_utility": v["models"][k]["u"],
            "findings": v["models"][k]["findings"],
        }
        for k in keys
    }
    return FinishVerdict(
        accepted=True,  # 恒真——holdout 读即终局，无拒收重试
        terminal_utility={
            "u": {k: v["models"][k]["u"] for k in keys},
            "rank_consistency": v["rank"],
            "report": v.get("report"),
            "steps": m["steps"],
        },
        deliverables=deliverables,
        decision_log=decision_log,
    )


def _final_flow_yaml(flow: list[dict], family: str, key: str, final_params: dict) -> str:
    """remote 终训 plan 序列化（前缀已由 _flow_prefix 产——uploads/cv.split 形态）。"""
    from mflowy.driver.config import NodeConf
    from mflowy.driver.serializer import flow_to_yaml

    model_key = f"model__{family}"
    flow.append({model_key: NodeConf(node_key=model_key, name=f"终局训练-{key}", params=dict(final_params))})
    return flow_to_yaml(tuple(flow))


def _final_train_test_remote(ss, m, family, key, *, outcome, scoring):
    """终训 remote 分支（#218 门拆除，#225 裁定 5）：await/fetch/find 模型步 run →
    holdout 指标解析（run metrics 展平面 ``{target}.{metric}`` 逆向，terminal_u 单源）
    + 副产物/模型镜像本地 run → (镜像 run_id, loader, holdout)。账本两分支同形
    （镜像 run = 本地库 run——下游 predict/统计/泰勒零分叉）。"""
    from . import session_remote
    from .job_provider import get_job_provider
    from .job_provider.remote import RemoteJobProvider
    from .model._pipeline import HOLDOUT_BYPRODUCTS, terminal_u
    from .model.artifacts import ModelLoader

    assert outcome.job_id and outcome.platform_experiment_id  # submitted 形态构造保证
    provider = get_job_provider()
    client = provider.client if isinstance(provider, RemoteJobProvider) else session_remote.platform_client()
    session_remote.await_job(client, outcome.job_id)
    unpacked = session_remote.fetch_workspace(client, outcome.job_id, outcome.platform_experiment_id, ss.dir / "jobs")
    face = session_remote.find_model_run(unpacked, family)
    metrics = session_remote.metrics_from_flat(face.client.get_run(face.run_id).data.metrics)
    if not metrics:
        raise RuntimeError(f"终训 remote run 无 holdout 指标（run={face.run_id[:8]}）——training 行为契约不符")
    holdout = {"u": round(terminal_u(metrics, scoring), 4), "metrics": metrics}
    from .model._pipeline import holdout_byproducts

    _byproducts = holdout_byproducts(TASKTYPE(m["task"]))
    for name in _byproducts.values():
        face.artifact(name)  # 副产物在场过门（任务感知——分类含 y_proba）
    y_pred_path = face.artifact(HOLDOUT_BYPRODUCTS["y_pred"])
    assert y_pred_path is not None  # 上行过门已抛——收窄
    pd.read_parquet(y_pred_path)  # 可读性过门（#236）
    model_dir = session_remote.logged_model_dir(face.client, unpacked, face.run_id)

    from mflowy.utils.mlflow import log_file

    with _session_run(m, f"final_train_test:{key}") as mirror:
        for name in _byproducts.values():
            log_file(face.artifact(name), artifact_file=name)
        import mlflow

        mlflow.log_artifacts(str(model_dir), artifact_path="final")
    loader = ModelLoader(name=family, _model_uri=f"runs:/{mirror.info.run_id}/final")
    loader.holdout = holdout
    return mirror.info.run_id, loader, holdout


def _flow_prefix(m: dict, fe: dict, *, file_path: str | None = None, split_params: dict | None = None) -> list[dict]:
    """plan 公共前缀（#218 抽取）：load__file → x_y → cv → [FE 节点]。

    - ``file_path=None`` → 会话源数据本径（local 寻址）；给定 → uploads 寻址
      （remote 容器 = ``/user-data/uploads/<basename>``，#225 实施细节）
    - ``split_params=None`` → ``cv__load``（from_run_id 按引用复用会话划分——终训
      形态）；给定 → ``cv__split`` 确定性复现（remote 形态：同节点同参数同缺省
      种子 = 同一划分——无引用可依时的等价数据面）"""
    from mflowy.driver.config import NodeConf
    from mflowy.model import registry

    if split_params is None and not m["split"].get("run_id"):
        raise RuntimeError("会话缺划分身份（split.run_id）——终训 plan 的 cv.load 引用源，开局物化异常")
    xy_params: dict = {"target": m["target"], "task": m["task"]}
    if m.get("id_columns"):
        xy_params["id_columns"] = list(m["id_columns"])
    flow: list[dict] = [
        {
            "load__file": NodeConf(
                node_key="load__file", name="导入冻结数据", params={"file_path": file_path or m["source_file"]}
            )
        },
        {"x_y": NodeConf(node_key="x_y", name="划分目标", params=xy_params)},
    ]
    if split_params is None:
        flow.append(
            {
                "cv__load": NodeConf(
                    node_key="cv__load",
                    name="引用会话划分（身份按 run 引用）",
                    params={"from_run_id": m["split"]["run_id"]},
                )
            }
        )
    else:
        flow.append(
            {
                "cv__split": NodeConf(
                    node_key="cv__split", name="确定性复现会话划分（同参数同种子）", params=dict(split_params)
                )
            }
        )
    if fe["cat"] in registry.FE_CAT_NODES:
        nk = f"x_transformer__{registry.FE_CAT_NODES[fe['cat']]}"
        flow.append({nk: NodeConf(node_key=nk, name="类别编码")})
    if fe["num"] in registry.FE_NUM_NODES:
        nk = f"x_transformer__{registry.FE_NUM_NODES[fe['num']]}"
        flow.append({nk: NodeConf(node_key=nk, name="数值缩放")})
    return flow


def _plan_yaml(m: dict, key: str, final_params: dict | None = None) -> str:
    """per-combo 终训计划（Conf → flow_to_yaml 单源序列化，可独立重放 = 那次终训
    本身）：load__file → x_y（target + id_columns 剔除——与开局同数据面）→ cv.load
    （from_run_id 按引用复用会话划分）→ [FE 节点] → 族节点（final_params 全 pin →
    空间空 → training 行为，确定性重放）。"""
    from mflowy.driver.config import NodeConf
    from mflowy.driver.serializer import flow_to_yaml

    entry = m["models"][key]
    flow = _flow_prefix(m, entry["fe"])
    params = dict(final_params if final_params is not None else (entry["final_params"] or {}))
    model_key = f"model__{entry['family']}"
    flow.append({model_key: NodeConf(node_key=model_key, name=f"终局训练-{key}", params=params)})
    return flow_to_yaml(tuple(flow))


def _hpo_plan_yaml(
    m: dict,
    fe: dict,
    *,
    family: str,
    n_trials: int,
    seed,
    scoring,
    inherit_study_run_id: str | None,
    remote_transport: bool,
    degenerate_dim: str | None = None,
) -> str:
    """run_hpo 搜索 plan（provider 全形态共用）：load → x_y → cv → [FE] → 族节点
    （search 行为：光标 H 空间/标量 + n_trials + scoring + seed +
    inherit_study_run_id）。remote_transport → uploads 寻址 + cv.split 确定性复现
    （无本地 run 引用可依）；否则本径 + cv.load 按引用复用会话划分。退化形
    （全 pin）：``degenerate_dim`` 以单值 DiscreteSpace 渲染——search 行为保真
    （inner CV 评估，不碰 holdout）。"""
    from mflowy.driver.config import NodeConf
    from mflowy.driver.serializer import flow_to_yaml
    from mflowy.model.model.study import ContinuousSpace, DiscreteSpace
    from mflowy.model.model.utils import space_from_json

    if remote_transport:
        from mflowy.utils.constants import REMOTE_UPLOADS_DIR

        split_params = {**m["protocol"]["split_params"]}
        split_params.setdefault("cv", m["protocol"]["cv"])  # 缺省单源（#83）：与开局物化同面
        flow = _flow_prefix(
            m, fe, file_path=f"{REMOTE_UPLOADS_DIR}/{Path(m['source_file']).name}", split_params=split_params
        )
    else:
        flow = _flow_prefix(m, fe)
    params: dict = {
        k: (space_from_json(v) if isinstance(v, dict) else v) for k, v in m["cursor"]["H"].items()
    }  # 空间槽位（JSON 形态）→ Space 本体（序列化回 YAML 构造形）
    if degenerate_dim is not None:  # 全 pin 退化形：单点空间——DAG 侧保持 search 行为
        v = params.get(degenerate_dim, _family_params(family)[degenerate_dim])  # pin 值或声明缺省
        declared = _family_params(family)[degenerate_dim]  # 空间类型随签名（float|Continuous vs int|Discrete）
        params[degenerate_dim] = (
            ContinuousSpace(float(v), float(v))
            if isinstance(declared, float)
            else (ContinuousSpace(v, v) if isinstance(declared, ContinuousSpace) else DiscreteSpace([v]))
        )
    params["n_trials"] = n_trials
    params["scoring"] = scoring
    if seed is not None:
        params["random_state"] = seed
    if inherit_study_run_id:
        params["inherit_study_run_id"] = inherit_study_run_id
    model_key = f"model__{family}"
    flow.append({model_key: NodeConf(node_key=model_key, name=f"会话搜索-{family}", params=params)})
    return flow_to_yaml(tuple(flow))


# ── observe：读时投影（π 的决策面——菜单退役，#117/#120） ───────────────────


def _space_summary(family: str) -> dict:
    """签名超参面摘要：动态维（Space 声明，含边界）vs 静态常量——π 的参数化依据。"""
    from mflowy.model.model.study import ContinuousSpace, DiscreteSpace
    from mflowy.model.model.utils import space_to_json

    dynamic, static = {}, {}
    for name, default in _family_params(family).items():
        if isinstance(default, (ContinuousSpace, DiscreteSpace)):
            dynamic[name] = space_to_json(default)
        else:
            static[name] = default
    return {"dynamic": dynamic, "static": static}


def _board(m: dict) -> dict:
    """账本板面（排序面；真源 = 盘上 study，条目值为派生缓存）。"""
    return {
        k: {
            "family": v["family"],
            "fe": v["fe"],
            "u_inner": v["u_inner"],
            "n_trials": v["n_trials_at_last_eval"],
            "std": v["std"],
            "frozen": v["frozen"],
            "u": v["u"],
        }
        for k, v in m["models"].items()
    }


def _digest(m: dict) -> str:
    """状态摘要指纹：预算消耗 + 光标 + 账本快照（马尔可夫性）。"""
    c = m["cursor"]
    board = ",".join(
        f"{k.split('|')[0]}:{v['u_inner']}{'✻' if v['frozen'] else ''}" for k, v in sorted(m["models"].items())
    )
    return (
        f"runs={m['steps']}/{m['max_steps']};m={c['m']};seed={c['seed']};fe={c['F']['cat']}/{c['F']['num']};"
        f"pins={sorted(c['H'])};board={board or 'empty'}"
    )


def save_loader(session_id: str, loader, key: str, v: int = 1) -> None:
    """final_train_test 后持久化 loader 门面（键 = 结构键 + _v{n} 冻结世代；
    多模型会话各持一份，会话内相对寻址）。

    M5（#89 ④片）：freeze-once（combo 冻结后不可再终训）⇒ 世代恒 1——版本号
    显式落名（规格字面：带版本文件名，与 data 域 v{n}.parquet 同纪律）；写走
    write_bytes_atomic（先 dumps 后落盘，红线 3）。
    """
    write_bytes_atomic(
        _root() / session_id / f"loader_{_key_file(key)}_v{v}.pkl",
        pickle.dumps(loader, protocol=pickle.HIGHEST_PROTOCOL),
    )


def load_loader(session_id: str, key: str, v: int = 1):
    with open(_root() / session_id / f"loader_{_key_file(key)}_v{v}.pkl", "rb") as f:
        return pickle.load(f)
