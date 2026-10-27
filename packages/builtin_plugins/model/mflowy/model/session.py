"""建模会话引擎（应用层，#89 会话基座上壳）：#62 回路与状态机——s 元组（光标 + 账本）+ 八动作面。

状态 ``s = (光标 m/H/seed/F, 账本 board, oof, budget,
done, history)``；动作面 ``A = {select_model, set_hp, set_hp_space, set_seed,
select_FE}（配置编辑：零步费零奖励）∪ {run_hpo, final_train_test}（计算动作：吃步
——步数口径经 counts_step 谓词声明给壳的预算门）∪ {done}``（= finish_session
控制面独立入口，同 clean 域）。

会话恒**单目标任务**（x_y target 收窄——多目标在会话外三层循环：多目标 →
多模型 → 单目标任务会话，文献综述阶段输出每目标的模型候选）。

光标 **单表 H**（#66 BREAKING）：``H[param]`` 唯一槽位——标量 = pin、dict
（space_to_json 形态）= 空间收窄；set_hp/set_hp_space 同写 H，last-write-wins
显式（decision_log note 记覆盖前后值）。

盘上（#89 壳三件 + 域状态/工件）：
- observation.json / decision.log / rules.pkl —— 壳三件（driver.session）；
  prior = 冻结协议（source_file/target/task/card/protocol/candidates/id_columns）
- board.json —— 域状态（光标/快照栈/账本/verdict 缓存）；写序 = board 先行 →
  日志追加（commit）→ 观测投影
- split.pkl / study_*.db / loader_*.pkl / plans/ / deliverables/ —— 工件

- **run_hpo**：sqlite study per (family,F) 增量续跑；reward = Δu_inner（新键 =
  u_inner）；账本条目随附证据配置快照（final_train_test 交付面）；findings 浮出
  进 state_digest（π 的过程决策面）
- **final_train_test**（#66 终训走 modeling 工具）：先搜后门店——渲染 per-combo
  计划（cv.load(from_run_id) 按引用 + 证据配置快照 ∪ θ* 全 pin，Conf →
  flow_to_yaml 单源序列化）→ JobProvider DAG 跑终训/终评/副产物 → u/metrics 回读
  → 副产物指针落账 → holdout 终评恰一次 → combo 冻结；reward = 0
- **finish(pdf)**（#66 单动作）：收编 π 产报告进 deliverables + verdict（幂等
  重导出由壳只读重入保证——双 commit 不可能）；素材面走 draw_taylor_diagram /
  modeling_report_skeleton 两工具
- **账本真源** = 盘上 sqlite study（append-only）；rollback 只回退光标——board
  永不与盘分叉（#58 P1 根治）
- **propose**：pull-only 单源谓词机器（类型可用(s) ∧ 参数合法(s,a)）——findings
  驱动层（规则引擎实例，salience 记分牌序）+ 基座谓词机层；findings 只对本 combo
  键出题（#66）；词表外动作由 apply 边界 fail-loud 拒绝
- 划分身份 = cv.split 副产物 split.pkl 的 run_id 引用（开局走节点中间件链条
  物化——mlflow 窗口/log_cv 留痕/print 捕获；终训 plan 以 cv.load 按引用复用）
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from mflowy.driver.pomdp import Activation, FinishVerdict, Observation
from mflowy.driver.session import POMDPSession
from mflowy.model import registry
from mflowy.model.model._pipeline import oof_predictions, u_inner_of
from mflowy.model.model.types import MetricName
from mflowy.model.model.utils import MACHINERY_SKIP, infer_metric, normalize_space, space_from_json, space_to_json
from mflowy.utils.file import write_text_atomic

logger = logging.getLogger(__name__)

# 八动作面（done = finish_session 控制面独立入口）；compute = 吃步（counts_step 口径）
ACTIONS = ("select_model", "set_hp", "set_hp_space", "set_seed", "select_FE", "run_hpo", "final_train_test")
COMPUTE_ACTIONS = frozenset({"run_hpo", "final_train_test"})
BOARD_FILE = "board.json"
DEFAULT_N_TRIALS = 10


def _root() -> Path:
    """会话 root（env MFLOWY_MODELING_SESSIONS，域自管）。"""
    return Path(os.environ.get("MFLOWY_MODELING_SESSIONS", str(Path.home() / ".mflowy" / "modeling_sessions")))


def _counts_step(action: str) -> bool:
    """步数口径（壳预算门的域声明）：预算 = 实验数——只有计算动作吃步。"""
    return action in COMPUTE_ACTIONS


def _open(session_id: str) -> POMDPSession:
    return POMDPSession.open(_root() / session_id, counts_step=_counts_step)


# ── 域状态文件（board.json）：光标/快照栈/账本/verdict 缓存 ──────────────────


def _board_load(ss: POMDPSession) -> dict:
    return json.loads((ss.dir / BOARD_FILE).read_text(encoding="utf-8"))


def _board_write(ss: POMDPSession, board: dict) -> None:
    write_text_atomic(str(ss.dir / BOARD_FILE), json.dumps(board, ensure_ascii=False, indent=1, default=str))


def _mview(o: Observation, board: dict) -> dict:
    """manifest 形状的域逻辑视图：prior（冻结协议）+ state_digest 投影 + board 引用。

    board 的 cursor/models/snapshots 以**引用**进视图——域逻辑经 m 的变更即 board
    的变更，commit 时一次落盘。"""
    p, d = o.prior or {}, o.state_digest
    return {
        "session_id": o.session_id,
        **p,
        "split": d.get("split", {}),
        "cursor": board["cursor"],
        "snapshots": board["snapshots"],
        "models": board["models"],
        "verdict": board.get("verdict"),
        "steps": o.steps,
        "max_steps": o.max_steps,
        "finished": o.finished,
        "abandoned": o.abandoned,
    }


def _digest_state(m: dict) -> dict:
    """state_digest 投影：划分身份 + 光标 + 账本板面 + 指纹（域逻辑读 m）。"""
    return {
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
    from mflowy.data.types import TASKTYPE
    from mflowy.driver.config import parse_enum

    return parse_enum(TASKTYPE, m["task"])


def _metric_of(m: dict) -> MetricName:
    return MetricName(m["protocol"]["metric"]) if m["protocol"].get("metric") else infer_metric(_task(m))


def _candidates(m: dict) -> list[str]:
    """候选面（族级）：卡 candidates（prior 结构化清单）∩ 已声明族目录。"""
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


def _signature_of(family: str):
    """族节点签名（约束单一真源——set_hp/set_hp_space 校验面）。"""
    import inspect

    return inspect.signature(registry.families()[family]["pyfunc"])


def _family_params(family: str) -> dict:
    """族签名用户超参面（plumbing 除外；random_state 走 set_seed 专用动作）。"""
    from mflowy.model.model.utils import signature_params

    return signature_params(registry.families()[family]["pyfunc"])


# 空间声明/持久化的单源件住 model/utils：normalize_space / space_to_json /
# space_from_json / MACHINERY_SKIP——session 不留第二份拷贝


# ── 开局：协议谈判在剧本层（model-brief / research-review）完成，cv_card 传入即物化冻结 ──


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


def start_session(
    file_path: str,
    cv_card: str,
    *,
    target: str | None = None,
    task: str | None = None,
    metric: str | None = None,
    candidates: list[dict] | None = None,
    max_steps: int | None = None,
    id_columns: list[str] | None = None,
    goal: str = "",
    external_baseline: str = "",
    card: dict | None = None,
) -> Observation:
    """开局建模会话（cv_card = cv.split 配置 yaml 明文，必填，唯一主通道）：
    D 物化冻结 + 光标就位。

    **显式参数 > card（DomainCard 背景）分层缺省**——合并结果冻结进 prior
    （episode 背景状态，冻结牙齿在壳）；两处同显式来源的冲突仍由校验门 fail-loud。
    改协议 = 新 episode（弃局重开）。
    """
    card = card or {}
    if not cv_card or not str(cv_card).strip():
        raise ValueError("cv_card 必填——cv.split 配置 yaml 明文（协议谈判在剧本层 model-brief 完成）")
    split_params = _parse_split_yaml(cv_card)
    target = target or card.get("target")
    if not target:
        raise ValueError("target 必填——显式参数或 card.target（无静默缺省）")
    task = task if task is not None else card.get("task")
    if not task:
        raise ValueError("task 必填（regression/classification）——显式参数或 card.task")
    id_cols = list(id_columns) if id_columns is not None else list(card.get("id_columns") or ())
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

    from mflowy.model import propose_rules

    ss = POMDPSession(_root(), *propose_rules.default_engine().rules.values(), prefix="ms_")
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
    _materialize(ss, m, df, task)
    with ss.start_ctx(
        max_steps=proto["max_steps"],
        prior={
            "source_file": m["source_file"],
            "target": m["target"],
            "task": m["task"],  # 物化后落定的实际 task（x_y 推断）
            "card": card,
            "protocol": m["protocol"],
            "candidates": m["candidates"],
            "id_columns": id_cols,
        },
    ) as o:
        _board_write(ss, board)  # 域状态先行（开局 commit = 观测落盘）
        o.state_digest = _digest_state(m)
    return o


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
            xy = xy_ctx.handler(xy_ctx, load=df, target=m["target"], task=task, id_columns=m["id_columns"])
            xy_ctx.result = xy  # 链直调不自动落 result（Context.run 才落）——log_cv 侧信道依赖显式播种
            # 缺省单源（#83）：物化折数取冻结协议值——不二落 cv.split 签名缺省（两活源即漂移面）
            split_kwargs = {**m["protocol"]["split_params"]}
            split_kwargs.setdefault("cv", m["protocol"]["cv"])
            spec = split_ctx.handler(split_ctx, x_y=xy, **split_kwargs)
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

    with open(download_run_artifact(split_ctx.run_id, "split.pkl"), "rb") as f:
        ts = pickle.load(f)  # 身份按引用：会话 split.pkl ≡ run 副产物（零二次物化）
    with open(ss.dir / "split.pkl", "wb") as f:
        pickle.dump(ts, f)
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


# ── apply：八动作面分派（clean 形状；终态/预算门在壳的 apply_ctx） ───────────


def apply_action(session_id: str, node: str, params: dict | None = None) -> Observation:
    """执行一个回合动作：node ∈ 八动作面词表（配置编辑 + 计算动作；done = finish）。

    配置编辑只改光标（零步费零奖励，生效于该键下一次 run_hpo）；计算动作吃步
    （counts_step 口径 → 壳预算门）。词表外（族名/FE 节点名直 apply 已随 #62
    退役）fail-loud 带词表清单。
    """
    params = dict(params or {})
    if node not in ACTIONS:
        raise ValueError(
            f"动作 `{node}` 不在词表（八动作面 = {list(ACTIONS)}；done = finish 控制面独立入口，"
            "rollback/finish/abandon 同为独立入口）"
        )
    ss = _open(session_id)
    with ss.apply_ctx(node) as (o, log):
        board = _board_load(ss)
        m = _mview(o, board)
        dispatch = {
            "select_model": _select_model,
            "set_hp": _set_hp,
            "set_hp_space": _set_hp_space,
            "set_seed": _set_seed,
            "select_FE": _select_fe,
            "run_hpo": _run_hpo,
            "final_train_test": _final_train_test,
        }
        dispatch[node](ss, o, m, board, log, params)
    return o


def _commit(ss, o, m, board, log, action: str, params: dict, note: str = "", reward=None, findings=None) -> None:
    """统一提交点：board 先行（域状态）→ 日志追加（commit）→ 观测投影更新（ctx exit 落盘）。"""
    if action in COMPUTE_ACTIONS:
        o.steps += 1
        m["steps"] = o.steps
    record = {
        "step": len(log) + 1,
        "action": action,
        "params": params,
        "steps": m["steps"],
        "budget_left": m["max_steps"] - m["steps"],
        "note": note,
    }
    _board_write(ss, board)
    log.append(record)
    o.reward = reward
    state = _digest_state(m)
    if findings is not None:
        state["findings"] = findings  # 计算动作的过程决策面浮出（π 读数）
    o.state_digest = state


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
    meta = registry.families()[family]["mdp_family"]
    return meta["machinery"], meta.get("fit_params")


def _effective_overrides(m: dict) -> dict:
    """光标有效参数覆盖（单表 H 解析）：dict = Space 收窄、标量 = pin。"""
    return {k: (space_from_json(v) if isinstance(v, dict) else v) for k, v in m["cursor"]["H"].items()}


def _ensure_structure(m: dict, family: str) -> dict:
    """结构相容守卫 + 最小装配（encode 族 × 类别列 × cat 空槽 → run 自带 ordinal）。"""
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


def _open_study(m: dict, key: str, objective_id: dict, scoring: MetricName, seed: int | None):
    """盘上 study 真源：目标同一性判据——空间变（收窄/pin/加深）增量继续，目标变（seed/scoring）重开。

    sampler 随 seed 建（#84：objective_id 已含 seed，sampler 种子与目标同一性同步——
    曾走 optuna 缺省未钉 TPESampler，θ* 每进程漂移致 findings 出题面抖动）。"""
    import optuna
    from optuna import samplers

    db = _root() / m["session_id"] / f"study_{_key_file(key)}.db"
    storage = f"sqlite:///{db}"
    direction = "maximize" if MetricName.higher_is_better(scoring) else "minimize"
    sampler = samplers.TPESampler(seed=seed)
    if db.exists():
        study = optuna.load_study(study_name=key, storage=storage, sampler=sampler)
        if study.user_attrs.get("objective") == objective_id:
            return study
        db.unlink()  # 目标变 → 重开（trial 清零；旧文件弃置——append-only 真源换新目标）
    study = optuna.create_study(
        study_name=key, storage=storage, direction=direction, load_if_exists=True, sampler=sampler
    )
    study.set_user_attr("objective", objective_id)
    return study


def _tag_run(run, m: dict, *, action: str, key: str, family: str) -> None:
    """mlflow 血缘认证（fluent 窗口内）：委托 _lineage_tags 单源。"""
    import mlflow

    mlflow.set_tags(_lineage_tags(m, action=action, key=key, family=family))


def _run_hpo(ss, o, m, board, log, params: dict) -> None:
    """run_hpo：增量搜索（inner CV）→ θ* 上 OOF 观察面 → 账本更新（reward = Δu_inner）。"""
    import mlflow
    from mflowy.model.model._pipeline import hyperparameters_search

    n_trials = params.get("n_trials", DEFAULT_N_TRIALS)
    unknown = set(params) - {"n_trials"}
    if unknown:
        raise ValueError(f"run_hpo 只收 n_trials（pin/收窄走 set_hp/set_hp_space），多余参数 {sorted(unknown)}")
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

    scoring = _metric_of(m)
    seed = m["cursor"]["seed"]
    ts = _load_split(m)
    ct = registry.build_ct(fe["cat"], fe["num"], (ts.X_train, ts.y_train, _task(m)))
    machinery, _ = _machinery_of(family)
    make_pipe, base, space = machinery(_task(m), ct, seed=seed, **_effective_overrides(m))
    if not space and n_trials > 1:
        raise ValueError("全 pin 候选（有效空间为空）走 n_trials=1 退化形——多余 trial 是同一参数的重复评估")

    objective_id = {"seed": seed if seed is not None else "default", "scoring": scoring.value}
    study = _open_study(m, key, objective_id, scoring, seed)
    with mlflow.start_run(run_name=f"run_hpo:{key}") as run:
        _tag_run(run, m, action="run_hpo", key=key, family=family)
        from mflowy.utils.capture import capture_prints

        # 通道纪律（M7）：直调 _pipeline 绕过 driver 节点的唯一捕获窗口——自建窗口，
        # 业务数据（分数摘要）已进 mlflow/账本，捕获输出转 logger（stderr 诊断面）
        with capture_prints() as buf:
            hyperparameters_search(
                space,
                scoring=scoring,
                n_trials=n_trials,
                split=ts,
                make_pipe=make_pipe,
                base_params=base,
                study=study,
                seed=seed,
            )
            theta = {**base, **study.best_trial.params}
            y_oof, y_proba = oof_predictions(ts, make_pipe, theta)
            runs_dir = ss.dir / "runs" / run.info.run_id
            runs_dir.mkdir(parents=True, exist_ok=True)
            y_oof.to_parquet(runs_dir / "oof_y_pred.parquet", index=True)
            # #64 OOF 图集：观察面指向 OOF 源（cross_val_predict × train 侧 y_true），挂 run 下
            from mflowy.model.report import oof_plot_album

            plots = oof_plot_album(m["task"], ts.y_train, y_oof, y_proba, runs_dir / "plots")
        captured = buf.getvalue().strip()
        if captured:
            logger.info("run_hpo[%s] 捕获输出：\n%s", key, captured)

    prev_u = entry.get("u_inner")
    new_u = u_inner_of(study, scoring)
    best = study.best_trial
    scores = best.user_attrs.get("scores", [])
    import numpy as np

    # #35 diagnose 接线点：OOF 观察面上的机械化诊断（零步数）——落 models[key] + run 响应浮出
    from mflowy.model.diagnose import diagnose_oof

    findings = [
        {"source": "rule", **f}
        for f in diagnose_oof(
            m["task"],
            ts.X_train,
            ts.y_train,
            y_oof[y_oof.columns[0]].to_numpy() if isinstance(y_oof, pd.DataFrame) else y_oof,
            best_params=dict(best.params),
            space=space,
            fold_scores=scores,
        )
    ]
    entry.update(
        {
            "study": f"study_{_key_file(key)}.db",
            "objective": objective_id,
            "n_trials_at_last_eval": len(study.trials),
            "u_inner": new_u,
            "best_params": dict(best.params),  # θ*（动态维；pin 常量在证据快照）
            # #66 证据配置快照：搜索时刻完整静态配置（默认 ∪ pin——θ* 之外维度的
            # 搜索时取值）；final_train_test 交付 = evidence ∪ θ*，搜后改 pin 不生效
            "evidence": {k: v for k, v in base.items() if k in _family_params(family)},
            "std": round(float(np.std(scores)), 4) if len(scores) else None,
            "oof": {
                "run_id": run.info.run_id,
                "y_pred": f"runs/{run.info.run_id}/oof_y_pred.parquet",
                "plots": plots,
            },
            # 规则 findings 随每次搜索刷新；LLM findings（#36）保留
            "findings": findings + [f for f in entry.get("findings", []) if f.get("source") == "llm"],
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
        {"n_trials": n_trials, "key": key},
        note=f"u_inner={new_u}（Δ={reward}）trials={len(study.trials)} oof_run={run.info.run_id[:8]}",
        reward=reward,
        findings=findings,  # run 响应浮出（π 的过程决策面 → state_digest["findings"]）
    )


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
    """
    if params:
        raise ValueError("final_train_test 无参数——θ* 恒从账本读（先搜后店：无搜索直训不合法）")
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

    plan_text = _plan_yaml(m, key, final_params)
    plan_path = ss.dir / "plans" / f"final_{_key_file(key)}.yaml"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(plan_path, plan_text)  # 审计工件（执行合同面是明文文本，M4 计划落盘留痕）

    from .job_provider import get_job_provider

    result = _run_bridge(
        get_job_provider().modeling(
            modeling_steps=plan_text,
            name=f"final_train_test:{m['session_id'][:6]}",
            desc=f"会话 {m['session_id']} combo {key} 终训（终训即 plan 执行）",
        )
    )
    if result.status != "finished":
        raise RuntimeError(f"终训 DAG 未完成（status={result.status}）：{result.error}")
    nodes = [r for r in result.runs if r.node == f"model__{family}"]
    node = next((r for r in nodes if r.status == "FINISHED" and r.run_id), None)
    if node is None:
        raise RuntimeError(f"终训 DAG 无完成的 model__{family} 节点（runs：{[(r.node, r.status) for r in nodes]}）")
    loader = node.value
    holdout = getattr(loader, "holdout", None)
    if not holdout:
        raise RuntimeError("终训产物缺 holdout 快照（ModelLoader.holdout）——provider 实现与契约不符，请反馈")

    # 副产物指针经 run artifacts 清单解析落账（s 自描述——board 可见 run 下有什么）
    from mflowy.model.model._pipeline import HOLDOUT_BYPRODUCTS
    from mflowy.utils.mlflow import download_run_artifact, run_artifacts

    arts = run_artifacts(node.run_id)
    missing = set(HOLDOUT_BYPRODUCTS.values()) - set(arts)
    if missing:
        raise RuntimeError(f"终训 run 缺副产物 {sorted(missing)}（run artifacts：{arts}）——training 行为契约不符")
    byproducts = dict(HOLDOUT_BYPRODUCTS)
    y_pred = pd.read_parquet(download_run_artifact(node.run_id, HOLDOUT_BYPRODUCTS["y_pred"]))

    # 血缘认证 post-hoc 补打（DAG run 窗口外）
    import mlflow

    for k, v in _lineage_tags(m, action="final_train_test", key=key, family=family).items():
        mlflow.MlflowClient().set_tag(node.run_id, k, v)

    # findings：会话 split（≡ DAG 按引用复用的划分）+ 副产物 y_pred + u_inner
    ts = _load_split(m)
    y_pred_df = pd.DataFrame(y_pred.to_numpy(), index=ts.y_test.index, columns=ts.y_test.columns)
    from mflowy.model.model._pipeline import holdout_findings

    findings = [{"source": "rule", **f} for f in holdout_findings(ts, y_pred_df, entry.get("u_inner"), holdout["u"])]

    save_loader(m["session_id"], loader, key)
    entry.update(
        {
            "frozen": True,
            "u": holdout["u"],
            "metrics": {t: {k.value: v for k, v in tm.items()} for t, tm in holdout["metrics"].items()},
            "findings": findings + [f for f in entry.get("findings", []) if f.get("source") == "llm"],
            "run_id": node.run_id,
            "final_params": final_params,  # 证据配置（= plan 全 pin 面——重放承诺与执行事实同一）
            "byproducts": byproducts,
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
        note=f"u={holdout['u']} combo 冻结 run={node.run_id[:8]}（终训即 plan：{plan_path.name}）",
        reward=0,
        findings=findings,
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
    }


# ── 控制面 ──


def report_findings(session_id: str, key: str, findings: list[dict]) -> Observation:
    """LLM findings 落诊（#36 控制面）：不可机械化诊断进 models[key]——schema 校验、
    决策日志审计、与规则 findings 均匀聚合（propose 消费面不区分来源）。

    schema = diagnose 族 ErrorReport 先例（kind/severity/evidence/message）+
    ``source`` 引擎保留键；kind ⊆ 已知词表（#35 定案——语言超纲即拒）。
    """
    ss = _open(session_id)
    with ss.apply_ctx("report_findings") as (o, log):
        board = _board_load(ss)
        m = _mview(o, board)
        if key not in m["models"]:
            raise ValueError(f"落诊键 `{key}` 不在账本（已训 combo：{sorted(m['models'])}）")
        validated = [_validate_llm_finding(f) for f in findings]

        entry = m["models"][key]
        entry["findings"] = [f for f in entry.get("findings", []) if f.get("source") == "rule"] + validated
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
    return o


def _validate_llm_finding(f: dict) -> dict:
    """schema 校验（fail-loud 带可用面）：kind ⊆ 已知词表；severity ∈ [0,1]；未知顶层键拒绝。"""
    from mflowy.model.propose_rules import KNOWN_FINDING_KINDS

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


def rollback(session_id: str) -> Observation:
    """回退光标到上一快照（决策日志追加 rollback 记录；预算不回补；账本/study 永不
    回退——盘上 append-only）。"""
    ss = _open(session_id)
    with ss.rollback_ctx() as (o, log):
        board = _board_load(ss)
        m = _mview(o, board)
        if len(m["snapshots"]) <= 1:
            raise RuntimeError("无历史版本可回退")
        snap = m["snapshots"].pop()
        m["cursor"] = {
            "m": snap["m"],
            "H": dict(snap["H"]),
            "seed": snap["seed"],
            "F": dict(snap["F"]),
        }
        board["cursor"] = m["cursor"]
        record = {
            "step": len(log) + 1,
            "action": "rollback",
            "params": {},
            "steps": m["steps"],
            "budget_left": m["max_steps"] - m["steps"],
            "note": f"光标弹栈（预算不回补：steps={m['steps']}/{m['max_steps']}；账本与盘上 study 一致）",
        }
        _board_write(ss, board)
        log.append(record)
        o.reward = None
        o.state_digest = _digest_state(m)
    return o


def abandon_session(session_id: str, reason: str | None = None) -> dict:
    """弃局终局：委托壳 abandon（幂等）——置 abandoned、记录进决策日志、保留
    决策日志与模型账本、不导出交付物；已收局不可弃（终态互斥）。"""
    return _open(session_id).abandon(reason=reason)


# ── finish（done）：一次结算、多模型交付 ────────────────────────────────────


def finish_session(session_id: str, *, pdf: str) -> FinishVerdict:
    """收局（done，#66 单动作——报告是 π 的产物，引擎只收编不编译）：

    收编 π 产报告进 deliverables（复制 + tmp/rename 原子落盘；文件缺失 fail-loud）
    + verdict（u/findings/rank 一致性）。收编不验内容——报告是给人看的便当，
    数字权威面是 verdict/board/盘上 study；重入 = 幂等重导出（壳只读重入保证
    双 commit 不可能；交付报告不可变更——重写须弃局重开）。素材面走
    draw_taylor_diagram / modeling_report_skeleton 两工具，finish 只做一次收编。

    accepted 恒真（holdout 读在 final_train_test 恰一次，finish 只结算）。
    """
    ss = _open(session_id)
    board = _board_load(ss)
    o = ss.observation
    if o.abandoned:
        raise RuntimeError("会话已弃局：不可收局（弃局不可复活，改协议重开走新会话）")
    if board.get("verdict"):  # 已结算：幂等重导出（交付报告不可变更——审计面）
        adopted = board["verdict"]["report"]["source_pdf"]
        if pdf != adopted:
            raise ValueError(f"已收局交付报告不可变更（审计面）——重写报告须弃局重开；已收编报告源：{adopted!r}")
        m = _mview(o, board)
        return _verdict_response(m)

    m = _mview(o, board)
    keys = sorted(k for k, v in m["models"].items() if v["frozen"])
    if not keys:
        raise RuntimeError("无冻结 combo 可交付——先 run_hpo + final_train_test 再收局")

    src = Path(pdf)
    if not src.is_file():
        raise ValueError(
            f"报告文件不存在：{pdf!r}（先 draw_taylor_diagram + modeling_report_skeleton 拿素材，π 产报告后带路径收局）"
        )
    report = _adopt_report(ss.dir, src)

    verdict_models = {k: {"u": m["models"][k]["u"], "findings": m["models"][k]["findings"]} for k in keys}
    with ss.finish_ctx() as (fo, log):
        board["verdict"] = {
            "deliver_keys": keys,
            "models": verdict_models,
            "rank": _rank_consistency(m, keys),
            "report": report,
        }
        fo.finished = True
        record = {
            "step": len(log) + 1,
            "action": "finish",
            "params": {"pdf": str(src)},
            "steps": fo.steps,
            "budget_left": fo.max_steps - fo.steps,
            "note": f"交付 {keys}；报告 {report['file']}",
        }
        _board_write(ss, board)  # 域状态先行（verdict 缓存）
        log.append(record)
    m = _mview(ss.observation, _board_load(ss))
    return _verdict_response(m)


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


def taylor_report(session_id: str, run_ids: list[str]) -> dict:
    """泰勒聚合门面（draw_taylor_diagram 工具面）：ledger 反查 label（combo key）
    + 终训面校验（fail-loud 带可用清单）+ 副产物 fan-in（taylor_byproducts 边函数
    ——y_test 逐位同一断言）+ 图落会话目录。"""
    from mflowy.model.report import taylor_for_runs

    ss = _open(session_id)
    m = _mview(ss.observation, _board_load(ss))
    rid_to_key = {v["run_id"]: k for k, v in m["models"].items() if v.get("run_id")}
    unknown = [r for r in run_ids if r not in rid_to_key]
    if unknown:
        raise ValueError(
            f"run_ids {unknown} 无终训副产物（不在账本终训面——先 final_train_test）——可用 run：{sorted(rid_to_key)}"
        )
    runs = [(rid_to_key[r], r) for r in run_ids]
    out = taylor_for_runs(runs, m["task"], ss.dir)
    return {**out, "figure": out["figure"] if out["figure"] is None else str(out["figure"])}


def report_skeleton(session_id: str) -> dict:
    """报告素材样板（#66 只读零副作用）：表格从账本渲染（数字抄写归引擎——π 不
    默写 u_inner/θ*/u/rank）+ 图片清单（各 combo 的 OOF 图集、终训副产物指针、
    泰勒图路径）+ 叙事槽样板（判断归 π）。"""
    ss = _open(session_id)
    m = _mview(ss.observation, _board_load(ss))
    d = ss.dir
    from mflowy.model.report import TAYLOR_FIGURE  # 图名单源——与写入方共享常量（#83）

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
                "oof_plots": [f"runs/{e['oof']['run_id']}/plots/{p}" for p in e["oof"].get("plots", [])],
                "holdout_byproducts": e.get("byproducts"),
                "findings": e.get("findings", []),
            }
        )
    return {
        "session_id": session_id,
        "tables": {"board": board_rows},
        "rank_consistency": _rank_consistency(m, frozen),
        "images": images,
        "taylor": taylor_fig,
        "narrative_slots": [
            {"slot": "为什么是这个模型", "hint": "对比 board 的 u_inner/u/rank 与泰勒图的误差结构，说明交付选择"},
            {
                "slot": "局限与风险",
                "hint": "消费 findings（overfit_gap/slice_collapse/error_concentrated）与 rank_consistency",
            },
            {"slot": "下一步建议", "hint": "回流清洗域候选 / 换 FE 组合 / 加深搜索的证据化建议"},
        ],
        "board": _board(m),
        "decision_log": list(ss.decision_log),
    }


def _rank_consistency(m: dict, keys: list[str]) -> float | None:
    """SH 赌注出账：交付集上 u_inner 与 u 的秩相关。样本 < 2 返回 None——不伪装。"""
    import scipy.stats as st

    pairs = [(m["models"][k]["u_inner"], m["models"][k]["u"]) for k in keys]
    if len(pairs) < 2:
        return None
    rho = st.spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
    return round(float(rho), 4) if rho == rho else None


def _verdict_response(m: dict) -> FinishVerdict:
    """FinishVerdict（从 board 的 verdict 缓存确定性重建——幂等重导出的来源）。"""
    v = m["verdict"]
    keys = v["deliver_keys"]
    deliverables = {
        k: {
            "plan_yaml": _plan_yaml(m, k),
            "loader": {"session_id": m["session_id"], "file": f"loader_{_key_file(k)}.pkl"},  # sid 相对寻址
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
        decision_log=list(_open(m["session_id"]).decision_log),
    )


def _plan_yaml(m: dict, key: str, final_params: dict | None = None) -> str:
    """per-combo 终训计划（Conf → flow_to_yaml 单源序列化，可独立重放 = 那次终训
    本身）：load__file → x_y（target + id_columns 剔除——与开局同数据面）→ cv.load
    （from_run_id 按引用复用会话划分）→ [FE 节点] → 族节点（final_params 全 pin →
    空间空 → training 行为，确定性重放）。"""
    from mflowy.driver.config import NodeConf
    from mflowy.driver.serializer import flow_to_yaml

    entry = m["models"][key]
    fe = entry["fe"]
    if not m["split"].get("run_id"):
        raise RuntimeError("会话缺划分身份（split.run_id）——终训 plan 的 cv.load 引用源，开局物化异常")
    xy_params: dict = {"target": m["target"], "task": m["task"]}
    if m.get("id_columns"):
        xy_params["id_columns"] = list(m["id_columns"])
    flow: list[dict] = [
        {"load__file": NodeConf(node_key="load__file", name="导入冻结数据", params={"file_path": m["source_file"]})},
        {"x_y": NodeConf(node_key="x_y", name="划分目标", params=xy_params)},
        {
            "cv__load": NodeConf(
                node_key="cv__load",
                name="引用会话划分（身份按 run 引用）",
                params={"from_run_id": m["split"]["run_id"]},
            )
        },
    ]
    if fe["cat"] in registry.FE_CAT_NODES:
        nk = f"x_transformer__{registry.FE_CAT_NODES[fe['cat']]}"
        flow.append({nk: NodeConf(node_key=nk, name="类别编码")})
    if fe["num"] in registry.FE_NUM_NODES:
        nk = f"x_transformer__{registry.FE_NUM_NODES[fe['num']]}"
        flow.append({nk: NodeConf(node_key=nk, name="数值缩放")})
    params = dict(final_params if final_params is not None else (entry["final_params"] or {}))
    model_key = f"model__{entry['family']}"
    flow.append({model_key: NodeConf(node_key=model_key, name=f"终局训练-{key}", params=params)})
    return flow_to_yaml(tuple(flow))


# ── 候选生成：单源谓词机器（类型可用(s) ∧ 参数合法(s,a)） ───────────────────


def propose_actions(session_id: str) -> list[Activation]:
    """菜单（pull-only，规则引擎实例统一入口）：findings 驱动层（salience 记分牌序
    浮于顶部）+ 基座谓词机层（类型可用(s) ∧ 参数合法(s,a)——策展序）。

    - findings 驱动：modelRuleEngine 默认实例注册族级/FE 规则（propose_rules）；
      会话实例经壳 propose_ctx（rules.pkl 持载自定义规则）；fold_std → 加深
      （n_trials 门控回归防线）；complexity_edge / overfit_gap → 正则收窄/pin；
      slice_collapse → 类别权重提示；error_concentrated → FE 换档；无 leader
      （S 无模型信息）时 FE 候选零出现（规则资格内联，非权重压序）
    - 基座层：select_model 候选（baseline 第一、未跑在前、已训未全冻结降级末位）、
      run_hpo 首搜、final_train_test 交付、配置编辑动作面
    - LLM findings（#36）与规则 findings 均匀聚合（同一 fire 入口，无来源区分）
    - 终态（收局/弃局）返回 []
    - 菜单合并（同 (action, params) 多命中浮出一项）留域——引擎不去重
    """
    ss = _open(session_id)
    if ss.is_terminal:
        return []
    with ss.propose_ctx() as (o, engine):
        board = _board_load(ss)
        m = _mview(o, board)

        fams = registry.families()
        cursor = m["cursor"]
        key = _structure_key(cursor["m"], cursor["F"]) if cursor["m"] else None
        entry = m["models"].get(key) if key else None

        # ── findings 驱动层 ──
        # #66 作用域 = 本 combo 键：findings（规则与 LLM 同粒度）只对本键出题——菜单
        # rationale 引用的证据不来自另一个 FE 组合（findings 随 combo 居住）
        findings = [{"key": k, **f} for k, e in m["models"].items() if k == key for f in (e.get("findings") or [])]
        fe_slots_ok = {"cat": False, "num": False}
        if cursor["m"]:
            st = fams[cursor["m"]]["mdp_family"]["structure"]
            fe_slots_ok = {"cat": st["cat_support"] != "native" and _has_cat(m), "num": bool(st["scaling_need"])}
        import inspect as _inspect

        ctx = {
            "family": cursor["m"],
            "has_model": bool(cursor["m"]),
            "entry": entry,
            "combo_key": key,
            "cursor": {k: (dict(v) if isinstance(v, dict) else v) for k, v in cursor.items()},
            "budget_left": m["max_steps"] - m["steps"],
            "signatures": {fam: _family_params(fam) for fam in fams},
            # 加深题回归防线的探测源：完整签名（plumbing skip 的 _family_params 不含 n_trials）
            "has_n_trials": {fam: "n_trials" in _inspect.signature(fams[fam]["pyfunc"]).parameters for fam in fams},
            "regularizers_of": {fam: tuple(getattr(fams[fam]["pyfunc"], "regularizers", ()) or ()) for fam in fams},
            "fe_slots_ok": fe_slots_ok,
        }
        # 触发证据写进观测（#89 ③片：规则纯消费 o——findings/菜单上下文 = 检测富化面）
        o.state_digest = {**o.state_digest, "findings": findings, "menu_ctx": ctx}
        engine.fire(o=o)
        activations = engine.poll_proposals()
        subscribed = {getattr(r, "kind", None) for r in engine.rules.values()}
        if unconsumed := sorted({f["kind"] for f in findings} - subscribed):
            logger.warning("findings 无规则订阅（无静默丢弃——补订阅或修 kind）：%s", unconsumed)
        driven = [
            {
                "rule_key": a.rule_key,
                "salience": a.salience,
                "action": a.action,
                "params": dict(a.params),
                "rationale": a.rationale,
                "evidence": {
                    **(a.evidence or {}),
                    "_rule": {"rule_key": a.rule_key, "kind": engine.kind_of(a.rule_key), "salience": a.salience},
                },
            }
            for a in activations
        ]

        # ── 基座谓词机层（策展序） ──
        proposals: list[dict] = []

        # select_model：未跑候选族在前（baseline 第一——research-review 锚点）
        by_baseline = {c["family"]: c for c in m.get("candidates") or []}
        ordered = sorted(_candidates(m), key=lambda f: (not by_baseline.get(f, {}).get("baseline", False), f))
        for fam in ordered:
            entries = [v for v in m["models"].values() if v["family"] == fam]
            if not entries:
                cand = by_baseline.get(fam, {})
                proposals.append(
                    {
                        "rule_key": "base.select_model",
                        "salience": -1,  # 基座层垫底（记分牌约定：规则驱动在前）
                        "action": "select_model",
                        "params": {"family": fam},
                        "rationale": (
                            "baseline 锚点候选——先跑它定参照系"
                            if cand.get("baseline")
                            else "未跑候选——先 select_model 定光标，再 run_hpo 出首条账本"
                        ),
                        "evidence": {
                            "space": _space_summary(fams[fam]["pyfunc"]),
                            "tried": False,
                            "baseline": bool(cand.get("baseline", False)),
                            "space_hints": cand.get("space_hints") or {},
                        },
                    }
                )
        # 已训未全冻结：降级末位（θ*/u_inner/u 并排）
        for fam in ordered:
            entries = [v for v in m["models"].values() if v["family"] == fam]
            if entries and not all(v["frozen"] for v in entries):
                best = max(entries, key=lambda v: v["u_inner"] if v["u_inner"] is not None else float("-inf"))
                proposals.append(
                    {
                        "rule_key": "base.select_model",
                        "salience": -1,
                        "action": "select_model",
                        "params": {"family": fam},
                        "rationale": f"已训族（u_inner={best['u_inner']}）——重选可加深搜索/换 FE 组合",
                        "evidence": {
                            "tried": True,
                            "u_inner": best["u_inner"],
                            "best_params": best["best_params"],
                            "u": best["u"],
                            "frozen_combos": sum(1 for v in entries if v["frozen"]),
                        },
                    }
                )

        if cursor["m"]:
            # run_hpo 首搜（已搜键的加深题走 findings 驱动层——fold_std 门控）
            if entry is None:
                proposals.append(
                    {
                        "rule_key": "base.run_hpo",
                        "salience": -1,
                        "action": "run_hpo",
                        "params": {"n_trials": DEFAULT_N_TRIALS},
                        "rationale": "首搜（trial 落盘 sqlite study 增量续跑）",
                        "evidence": {"key": key, "space": _effective_space_summary(m), "n_trials": DEFAULT_N_TRIALS},
                    }
                )
            elif entry.get("u_inner") is not None and not entry["frozen"]:
                proposals.append(
                    {
                        "rule_key": "base.final_train_test",
                        "salience": -1,
                        "action": "final_train_test",
                        "params": {},
                        "rationale": f"终训交付（θ* 已在账本：{entry['best_params']}）——holdout 终评恰一次后 combo 冻结",
                        "evidence": {
                            "key": key,
                            "u_inner": entry["u_inner"],
                            "n_trials": entry["n_trials_at_last_eval"],
                        },
                    }
                )
            proposals.append(
                {
                    "rule_key": "base.set_hp",
                    "salience": -1,
                    "action": "set_hp",
                    "params": {},
                    "rationale": "pin 常量（生效于下一次 run_hpo；静态常量可 pin、动态维可收窄）",
                    "evidence": {"face": _pin_face(m)},
                }
            )
            proposals.append(
                {
                    "rule_key": "base.set_hp_space",
                    "salience": -1,
                    "action": "set_hp_space",
                    "params": {},
                    "rationale": "收窄动态维（空间变增量继续；seed/scoring 变 study 重开）",
                    "evidence": {"face": _space_face(m)},
                }
            )
            proposals.append(
                {
                    "rule_key": "base.set_seed",
                    "salience": -1,
                    "action": "set_seed",
                    "params": {},
                    "rationale": "设随机种子（搜索的环境参数——换 seed = 换目标 = study 重开）",
                    "evidence": {"current": cursor["seed"]},
                }
            )
            # select_FE 基座档：有模型账本信息（leader）才出现——S 无模型信息时 FE
            # 候选零出现（规则资格内联早退，非记分压序；动作边界只看 cursor.m——菜单是建议不是边界）
            if m["models"]:
                _fe_base_entries(m, cursor, fams, proposals)
        merged = [menu_activation(c) for c in _merge_menu(driven + proposals)]
        stop = [a for a in merged if a.action == "finish"]
        return [a for a in merged if a.action != "finish"] + stop  # 停止建议恒居末


def menu_activation(c: dict) -> Activation:
    """菜单项 → Activation（Handle 契约面）：基座层 rule_key=base.*、salience=-1 垫底。"""
    return Activation(
        rule_key=c.get("rule_key") or f"base.{c['action']}",
        action=c["action"],
        params=dict(c.get("params") or {}),
        salience=c.get("salience", -1),
        rationale=c.get("rationale") or "",
        evidence=dict(c.get("evidence") or {}),
    )


def _merge_menu(items: list[dict]) -> list[dict]:
    """(action, params) 键合并（#38 会话侧补位——引擎不去重）：同动作同参 = 同一
    建议的多重证据，浮出一项——rationale 并集（保序去重、"；"连接）；命中的规则
    溯源并列进 ``evidence["co_hits"]``（每条含 rationale 与 ``_rule`` 分解——合并
    不静默）。记分牌位保首现实体（poll 已按 salience 分排序——排序
    确定性与可审计不变）。
    """
    out: list[dict] = []
    pos: dict[tuple[str, str], int] = {}
    for c in items:
        k = (c["action"], _json(dict(c.get("params") or {})))
        if k not in pos:
            pos[k] = len(out)
            entry = dict(c)
            entry["_rats"] = [c["rationale"]] if c.get("rationale") else []
            entry["_hits"] = (
                [{"rationale": c["rationale"], "_rule": c["evidence"]["_rule"]}]
                if c.get("evidence", {}).get("_rule")
                else []
            )
            out.append(entry)
            continue
        cur = out[pos[k]]
        if c.get("rationale") and c["rationale"] not in cur["_rats"]:
            cur["_rats"].append(c["rationale"])
        rule = c.get("evidence", {}).get("_rule")
        if rule:
            cur["_hits"].append({"rationale": c["rationale"], "_rule": rule})
    for entry in out:
        merged = len(entry["_rats"]) > 1 or len(entry["_hits"]) > 1
        if len(entry["_rats"]) > 1:
            entry["rationale"] = "；".join(entry["_rats"])
        if merged and entry["_hits"]:
            entry.setdefault("evidence", {})["co_hits"] = entry["_hits"]
        del entry["_rats"], entry["_hits"]
    return out


def _fe_base_entries(m: dict, cursor: dict, fams: dict, proposals: list) -> None:
    st = fams[cursor["m"]]["mdp_family"]["structure"]
    for slot, nodes_map in (("cat", registry.FE_CAT_NODES), ("num", registry.FE_NUM_NODES)):
        if slot == "cat" and (st["cat_support"] == "native" or not _has_cat(m)):
            continue
        if slot == "num" and not st["scaling_need"]:
            continue
        for option, node in sorted(nodes_map.items()):
            if option == cursor["F"][slot]:
                continue
            proposals.append(
                {
                    "rule_key": "base.select_FE",
                    "salience": -1,
                    "action": "select_FE",
                    "params": {"node": node},
                    "rationale": f"{cursor['m']} 的 {slot} 槽 {cursor['F'][slot]} → {option}（换 FE 后 run_hpo 出新账本条目）",
                    "evidence": {"slot": slot, "current": cursor["F"][slot], "structure": st},
                }
            )


def _pin_face(m: dict) -> dict:
    """set_hp 参数面：签名超参（当前槽位状态 + 静态/动态分型——单表 H 直读）。"""
    face = _family_params(m["cursor"]["m"])
    return {
        "params": {
            k: {"current_pin": v if not isinstance(v, dict) else None, "default": _json_default(d)}
            for k, d in sorted(face.items())
            if not _is_space(d)
            for v in [m["cursor"]["H"].get(k)]
        }
    }


def _space_face(m: dict) -> dict:
    face = _family_params(m["cursor"]["m"])
    return {
        "params": {
            k: space_to_json(d) | {"current_narrow": v if isinstance(v, dict) else None}
            for k, d in sorted(face.items())
            if _is_space(d)
            for v in [m["cursor"]["H"].get(k)]
        }
    }


def _effective_space_summary(m: dict) -> dict:
    """有效空间摘要：签名空间经光标收窄后的实况。"""
    face = _family_params(m["cursor"]["m"])
    out = {}
    for k, v in face.items():
        if _is_space(v):
            cur = m["cursor"]["H"].get(k)
            out[k] = cur if isinstance(cur, dict) else space_to_json(v)
    return out


def _is_space(v) -> bool:
    from mflowy.model.model.study import ContinuousSpace, DiscreteSpace

    return isinstance(v, (ContinuousSpace, DiscreteSpace))


def _json_default(v):
    if _is_space(v):
        return space_to_json(v)
    return v


def _space_summary(fn) -> dict:
    """签名空间摘要：动态维（Space 声明，含边界）vs 静态常量——超参面对 agent
    浮出（黑盒开箱）；plumbing 参数不入摘要。"""
    from mflowy.model.model.utils import signature_params

    dynamic, static = {}, {}
    for name, default in signature_params(fn).items():
        if _is_space(default):
            dynamic[name] = space_to_json(default)
        else:
            static[name] = default
    return {"dynamic": dynamic, "static": static}


# ── 观测投影（域逻辑的 m 视图 → state_digest 形状） ─────────────────────────


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


def _json(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def save_loader(session_id: str, loader, key: str) -> None:
    """final_train_test 后持久化 loader 门面（键 = 结构键；多模型会话各持一份，sid 相对寻址）。

    M5（#89 ④片）：freeze-once（combo 冻结后不可再终训）⇒ 每 key 恰写一次——
    文件名即版本（无覆写路径）；写走 write_bytes_atomic（先 dumps 后落盘，
    红线 3）。
    """
    from mflowy.utils.file import write_bytes_atomic

    write_bytes_atomic(
        str(_root() / session_id / f"loader_{_key_file(key)}.pkl"),
        pickle.dumps(loader, protocol=pickle.HIGHEST_PROTOCOL),
    )


def load_loader(session_id: str, key: str):
    with open(_root() / session_id / f"loader_{_key_file(key)}.pkl", "rb") as f:
        return pickle.load(f)
