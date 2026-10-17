"""建模会话引擎（应用层）：真决策 MDP——π 在调用方 LLM，apply(node, params) 与 clean 域同形。

#31 + grilling 定案（SH 依据两段式取代 P1-P3 相位机/board 词表/edit 动作）：

- 动作空间 A = **model 族节点**（run 语义：search → θ* + u_inner → 终训 → loader 落盘）∪ **FE 节点**（pipeline 成员语义：设置/替换模型前的
  FE 槽位；命中规则 = ①模型已选定 ②不在 leader ``structure`` 声明的冗余范围）
  ∪ 控制面（rollback/finish/abandon 独立函数）。"run" 是域语义名而非词表项。
- S = pipeline 成员及各成员参数（``pipeline.fe`` 槽位 + ``models``）+ 单一预算
  + 决策日志；board 由 models 派生（键 = (family, FE 组合)——不同 FE 下成绩
  并排可比）；FE 动作吃步数（同槽覆盖换档，清槽走 rollback）。
- 不变量：候选已声明才可 run（research-review 先验第一轮淘汰）；no-peeking：
  run 全程不碰 holdout、models 只存 u_inner；test 读只在 finish 每交付模型恰一次
  （accepted 恒真无拒收；重入幂等重导出缓存 verdict，换交付集 loud 拒绝）。
  存储复用 #29 SessionStore（env MFLOWY_MODELING_SESSIONS）。
"""

from __future__ import annotations

import json
import os
import pickle
import uuid
from dataclasses import replace
from pathlib import Path

import pandas as pd
from mflowy.driver.discover import get_node
from mflowy.driver.mdp_session import SessionStore
from mflowy.model import registry
from mflowy.model.model._split import materialize_split
from mflowy.model.model.types import MetricName
from mflowy.model.model.utils import infer_metric


def _store() -> SessionStore:
    """会话存储基座（#29 SessionStore）：root 解析留域（env MFLOWY_MODELING_SESSIONS）。"""
    root = os.environ.get("MFLOWY_MODELING_SESSIONS", str(Path.home() / ".mflowy" / "modeling_sessions"))
    return SessionStore(root)


def _is_terminal(m: dict) -> bool:
    return bool(m.get("finished") or m.get("abandoned"))


def _load_split(m: dict):
    """从会话目录重建 TrainSplit（D 冻结的数据面；spec 含 inner splitter，整体 pickle）。"""
    with open(_store().session_dir(m["session_id"]) / "split.pkl", "rb") as f:
        return pickle.load(f)


def _split_for_run(m: dict):
    """run 用的数据面：id 列剔除（防 ID 泄漏，问卷先验）。"""
    ts = _load_split(m)
    id_cols = [c for c in m.get("id_columns", []) if c in ts.X_train.columns]
    if id_cols:
        ts = replace(ts, X_train=ts.X_train.drop(columns=id_cols), X_test=ts.X_test.drop(columns=id_cols))
    return ts


def _task(m: dict):
    from mflowy.data.types import TASKTYPE
    from mflowy.driver.config import parse_enum

    return parse_enum(TASKTYPE, m["task"])


def _metric_of(m: dict) -> MetricName:
    return MetricName(m["protocol"]["metric"]) if m["protocol"].get("metric") else infer_metric(_task(m))


def _candidates(m: dict) -> list[str]:
    """候选面（族级）：协议 candidate_families（空 = 全部已声明族）∩ 已声明族目录。"""
    fams = registry.families()
    declared = m["protocol"].get("candidate_families") or list(fams)
    return sorted(f for f in declared if f in fams)


def _has_cat(m: dict) -> bool:
    X = _load_split(m).X_train
    return any(not pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)


def _structure_key(family: str, fe: dict) -> str:
    """board/交付键 = (family, FE 组合)——不同 FE 下成绩并排可比。"""
    return f"{family}|{fe['cat']}|{fe['num']}"


def _key_file(key: str) -> str:
    return key.replace("|", "__")


def _board(m: dict) -> dict:
    """board 由 models 派生（排序面；不落盘——单一真源在 models）。"""
    return {k: {"u_inner": v["u_inner"], "n_trials": v["n_trials"]} for k, v in m["models"].items()}


def _leader(m: dict) -> dict | None:
    """board leader（u_inner 最高且非 None 的结构条目）——FE 冗余判定与菜单的对象。"""
    viable = sorted(
        ((v["u_inner"], k) for k, v in m["models"].items() if v["u_inner"] is not None), key=lambda t: -t[0]
    )
    return m["models"][viable[0][1]] if viable else None


def _digest(m: dict) -> str:
    """状态摘要指纹：预算消耗 + pipeline FE + board 快照（马尔可夫性）。"""
    fe = m["pipeline"]["fe"]
    board = ",".join(f"{k.split('|')[0]}:{v['u_inner']}" for k, v in sorted(m["models"].items()))
    return f"runs={m['steps']}/{m['max_steps']};fe={fe['cat']}/{fe['num']};board={board or 'empty'}"


# ── 开局：协议谈判在剧本层（model-brief / research-review）完成，ProtocolCard 传入即物化冻结 ──


def start_session(
    file_path: str, card: dict | None = None, protocol_card: dict | None = None, max_steps: int | None = None
) -> dict:
    """开局建模会话（protocol_card 必填，唯一主通道）：D 物化冻结 + 候选面就位。

    协议谈判在剧本层完成——episode = (冻结 D, 冻结协议, 单一预算)，改协议 =
    新 episode（弃局重开）。
    """
    card = card or {}
    if not protocol_card:
        raise ValueError("protocol_card 必填——协议谈判在剧本层（model-brief）完成，引擎不再提供无协议开局")
    df = pd.read_parquet(file_path) if file_path.endswith(".parquet") else pd.read_csv(file_path)
    target = card.get("target") or protocol_card.get("target")

    # 四类相容校验门：task↔dtype/metric↔task/cv↔数据/预算与先验
    from mflowy.model.protocol import ProtocolCard

    proto = ProtocolCard.build(
        df,
        target=target,
        **{k: v for k, v in protocol_card.items() if k not in ("target", "exploratory")},
    ).as_dict()

    session_id = f"ms_{uuid.uuid4().hex[:10]}"
    store = _store()
    d = store.root / session_id
    # 不走 store.create()：manifest 落盘是 commit 点，须在 split.pkl 之后
    d.mkdir(parents=True, exist_ok=False)

    fe = {"cat": "passthrough", "num": "passthrough"}
    manifest = {
        "session_id": session_id,
        "source_file": str(file_path),
        "target": target,
        "card": card,
        "protocol": {},
        "task": None,
        "id_columns": list(card.get("id_columns") or ()),
        # S = pipeline 成员及参数：每步（run/FE）前压栈，rollback 弹栈（预算不回补）
        "pipeline": {"fe": dict(fe)},
        "snapshots": [{"pipeline": {"fe": dict(fe)}, "models": {}}],
        "models": {},
        "steps": 0,
        "max_steps": int(max_steps or proto["max_steps"]),
        "decision_log": [],
        "verdict": None,
        "finished": False,
    }
    _materialize(manifest, df, proto)
    store.write_manifest(session_id, manifest)
    return _observe(manifest)


def _materialize(m: dict, df: pd.DataFrame, protocol: dict) -> None:
    """D 物化冻结（x_y + split + split.pkl）。"""
    from mflowy.data.types import TASKTYPE as _TT
    from mflowy.driver.config import parse_enum

    target = m["target"]
    raw_task = str(protocol.get("task", "")).strip()
    if raw_task.upper() in _TT.__members__:
        raw_task = raw_task.upper()
    task = parse_enum(_TT, raw_task) if raw_task else None

    x_y_fn = get_node("x_y", raw=True)
    xy = x_y_fn(load=df, targets=target, task=task)
    split_fn = get_node("split", raw=True)
    spec = split_fn(x_y=xy, cv=protocol.get("cv", 5))
    ts = materialize_split(spec, xy)
    with open(_store().session_dir(m["session_id"]) / "split.pkl", "wb") as f:
        pickle.dump(ts, f)
    m["task"] = task.value
    m["protocol"] = protocol


# ── apply(node, params)：clean 形状分派（model 族 → run / FE 节点 → 槽位） ──


def apply_action(session_id: str, node: str, params: dict | None = None) -> dict:
    """执行一个回合动作（与 clean_apply 同形：node ∈ 本域动作词表）。

    词表 = model 族节点（.mdp_family 声明者，run 语义）∪ FE 节点（x_transformer
    词表投影，pipeline 成员语义）。params 对 run = pin 常量/收窄 range + 引擎旋钮
    pin/收窄（未知参数即 TypeError——签名即约束；资源 = 签名参数，#33 全退）；对 FE = 空。
    """
    params = dict(params or {})
    store = _store()
    m = store.read_manifest(session_id)
    if _is_terminal(m):
        raise RuntimeError("会话已终态（收局/弃局）：动作面为空")

    fams = registry.families()
    if node in fams:
        return _run_model(store, m, node, params)
    if node in registry.fe_vocab():
        return _apply_fe(store, m, node, params)
    raise ValueError(
        f"动作 `{node}` 不在词表（model 族 ∪ FE 节点 = {sorted(fams)} ∪ {sorted(registry.fe_vocab())}；"
        "控制面 rollback/finish/abandon 为独立入口）"
    )


def _check_budget(m: dict) -> None:
    if m["steps"] >= m["max_steps"]:
        raise RuntimeError(f"预算耗尽（max_steps={m['max_steps']}）：请 finish 或 abandon")


def _push_snapshot(m: dict) -> None:
    m["snapshots"].append(
        {"pipeline": {"fe": dict(m["pipeline"]["fe"])}, "models": {k: dict(v) for k, v in m["models"].items()}}
    )


def _run_model(store, m: dict, family: str, params: dict) -> dict:
    """run 一体化（SH 两段式）：守卫（候选/预算/结构相容）→ 转移 → 观测 → 决策日志。"""
    if family not in _candidates(m):
        raise ValueError(f"候选 `{family}` 未在协议 candidate_families 声明（开局先验：{_candidates(m)}）")
    _check_budget(m)

    fams = registry.families()
    scoring = params.pop("scoring", None) or _metric_of(m)  # 协议 metric 单口径贯穿 run/finish

    fe = m["pipeline"]["fe"]
    structure = fams[family]["mdp_family"]["structure"]
    if not registry.valid_structure((fe["cat"], fe["num"], family), _has_cat(m), fams):
        # 最小装配：encode 族 × 有类别列 × cat 槽空——run 自带 ordinal（先定模型的
        # 前提），π 后续用 FE 动作换档；装配随步原子（rollback 回 models/账本，
        # 装配保留——它是本步的一部分）
        if _has_cat(m) and fe["cat"] == "passthrough" and structure["cat_support"] == "encode":
            m["pipeline"]["fe"]["cat"] = "ordinal"
            fe = m["pipeline"]["fe"]
        else:
            raise ValueError(
                f"pipeline FE {fe['cat']}/{fe['num']} 与族 `{family}` 结构不相容（相容性 = .mdp_family.structure）"
            )

    ts = _split_for_run(m)
    ct = registry.build_ct(fe["cat"], fe["num"], (ts.X_train, ts.y_train, _task(m)))

    pins = dict(params)  # 其余全部直传节点签名（pin/收窄/n_trials）——未知即 TypeError
    loader = fams[family]["pyfunc"](
        train_split=ts,
        x_preprocessors=ct,
        evaluate_holdout=False,
        scoring=scoring,
        **pins,
    )
    key = _structure_key(family, fe)
    save_loader(m["session_id"], loader, key)

    _push_snapshot(m)
    search = loader.search or {}
    m["models"][key] = {
        "key": key,
        "family": family,
        "u_inner": loader.u_inner,
        "n_trials": pins.get("n_trials"),
        "fe": dict(fe),
        "best_params": search.get("best_params"),
        "std": search.get("std"),
        # plan 源：pin 常量 + θ*（资源轴/full 由 _plan_yaml 注入）——重放 ≡ 终态
        "plan_params": {**pins, **(search.get("best_params") or {})},
    }
    m["steps"] += 1
    _log(
        m,
        family,
        dict(params),
        note=f"u_inner={loader.u_inner} fe={fe['cat']}/{fe['num']}",
    )
    store.write_manifest(m["session_id"], m)
    return _observe(m, reward=loader.u_inner)


def _apply_fe(store, m: dict, node: str, params: dict) -> dict:
    """FE 动作：设置/覆盖 pipeline 槽位（命中规则 = 模型已选定 + 非冗余；同槽换档覆盖）。"""
    if params:
        raise ValueError("FE 节点参数暂不支持——词表投影走默认档（自由参数化在模型超参面）")
    _check_budget(m)
    if not m["models"]:
        raise RuntimeError("FE 之前先选定模型——run 一个候选（FE 候选在模型候选之后出现）")

    leader = _leader(m)
    st = registry.families()[leader["family"]]["mdp_family"]["structure"]
    slot = registry.fe_vocab()[node]
    option = (
        {v: k for k, v in registry.FE_CAT_NODES.items()}[node]
        if slot == "cat"
        else {v: k for k, v in registry.FE_NUM_NODES.items()}[node]
    )
    if slot == "cat":
        if st["cat_support"] == "native":
            raise ValueError(f"leader {leader['family']} 原生类别支持（cat_support=native）——cat 编码冗余")
        if not _has_cat(m):
            raise ValueError("数据无类别列——cat 槽无对象")
    elif not st["scaling_need"]:
        raise ValueError(f"leader {leader['family']} 不需缩放（scaling_need=False）——num 槽冗余")

    cur = m["pipeline"]["fe"][slot]
    if cur == option:
        raise ValueError(f"{slot} 槽已是 {option}——同槽换档 apply 另一 FE 节点，清槽走 rollback")

    _push_snapshot(m)
    m["pipeline"]["fe"][slot] = option
    m["steps"] += 1
    _log(
        m,
        node,
        {"slot": slot},
        note=f"{slot} 槽 {cur} → {option}（leader={leader['family']}，换 FE 后重跑出新板面条目）",
    )
    store.write_manifest(m["session_id"], m)
    return _observe(m)


def _log(m: dict, action: str, params: dict, note: str = "") -> None:
    m["decision_log"].append(
        {
            "step": len(m["decision_log"]) + 1,
            "action": action,
            "params": params,
            "steps": m["steps"],
            "budget_left": m["max_steps"] - m["steps"],
            "note": note,
        }
    )


# ── 控制面 ──


def rollback(session_id: str) -> dict:
    """回退 pipeline.fe + models 到上一快照（决策日志保留；预算不回补——steps 现值保留）。"""
    store = _store()
    m = store.read_manifest(session_id)
    if _is_terminal(m):
        raise RuntimeError("会话已终态（收局/弃局）：不可回滚——决策日志与模型账本是审计面")
    if len(m["snapshots"]) <= 1:
        raise RuntimeError("无历史版本可回退")
    snap = m["snapshots"].pop()
    m["pipeline"], m["models"] = snap["pipeline"], snap["models"]
    _log(m, "rollback", {}, note=f"弹栈（预算不回补：steps={m['steps']}/{m['max_steps']}）")
    store.write_manifest(session_id, m)
    return _observe(m)


def abandon_session(session_id: str, reason: str | None = None) -> dict:
    """弃局终局：委托 #29 SessionStore.abandon——置 abandoned、reason 落盘（审计）、
    保留决策日志与模型账本、不导出交付物；已终态再弃即 raise。"""
    return _store().abandon(session_id, reason=reason)


# ── finish：一次结算、多模型交付 ──


def finish_session(session_id: str, deliver_keys: list[str] | None = None, preview: bool = False) -> dict:
    """收局：每交付模型 holdout 评估恰一次 → terminal_utility + findings + rank 一致性
    诊断 + per-model 可重放 plan_yaml。

    - ``deliver_keys``：交付集选择（缺省 = 全部已训练结构，键 = family|cat|num）
    - ``preview=True`` 只读探针：只看 inner 证据，不碰 holdout、零 mutation
    - accepted 恒真（test 读即终局，无拒收重试）；重入 = 幂等重导出缓存 verdict
      （换交付集 loud 拒绝——重选交付集须弃局重开）
    """
    from mflowy.model.model._pipeline import holdout_report

    store = _store()
    m = store.read_manifest(session_id)
    if m.get("abandoned"):
        raise RuntimeError("会话已弃局：不可收局（弃局不可复活，改协议重开走新会话）")

    if preview:
        models = {k: {f: v[f] for f in ("u_inner", "n_trials", "std", "best_params")} for k, v in m["models"].items()}
        return {
            "session_id": session_id,
            "preview": True,
            "board": _board(m),
            "pipeline": {"fe": dict(m["pipeline"]["fe"])},
            "models": models,
            "steps": m["steps"],
            "max_steps": m["max_steps"],
            "decision_log": m["decision_log"],
        }

    if m.get("verdict"):  # 已结算：幂等重导出（缓存 verdict，holdout 不二次读）
        if deliver_keys is not None and sorted(deliver_keys) != sorted(m["verdict"]["deliver_keys"]):
            raise ValueError(
                f"已结算交付集 {m['verdict']['deliver_keys']} 不可变更（test 读即终局）——重选交付集须弃局重开"
            )
        return _verdict_response(m)

    keys = list(deliver_keys) if deliver_keys else sorted(m["models"])
    if not keys:
        raise RuntimeError("无已训练模型可交付——先 run 候选再收局")
    unknown = [k for k in keys if k not in m["models"]]
    if unknown:
        raise ValueError(f"交付键 {unknown} 不在已训练结构 {sorted(m['models'])} 中")

    ts = _split_for_run(m)
    metric = _metric_of(m)
    verdict_models: dict[str, dict] = {}
    for key in keys:
        rep = holdout_report(load_loader(session_id, key), ts, metric)
        verdict_models[key] = {"u_full": rep["u_full"], "findings": rep["findings"]}

    rank = _rank_consistency(m, keys, verdict_models)
    # verdict notes：error_concentrated → 回流清洗域提示（建议式，不做会话内动作）
    notes = [
        f"误差集中列 `{f['evidence']['column']}`——回流清洗域修列（该列质量问题害了模型）"
        for k in keys
        for f in verdict_models[k]["findings"]
        if f["kind"] == "error_concentrated_column"
    ]
    m["finished"] = True
    m["verdict"] = {"deliver_keys": keys, "models": verdict_models, "rank": rank, "notes": notes}
    store.write_manifest(session_id, m)
    return _verdict_response(m)


def _rank_consistency(m: dict, keys: list[str], verdict_models: dict) -> float | None:
    """SH 赌注出账：交付集上 u_inner 与 u_full 的秩相关。样本 < 2 返回 None——不伪装。"""
    import scipy.stats as st

    pairs = [
        (m["models"][k]["u_inner"], verdict_models[k]["u_full"]) for k in keys if m["models"][k]["u_inner"] is not None
    ]
    if len(pairs) < 2:
        return None
    rho = st.spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
    return round(float(rho), 4) if rho == rho else None


def _verdict_response(m: dict) -> dict:
    """FinishVerdict 响应（从缓存 verdict 确定性重建——幂等重导出的来源）。"""
    v = m["verdict"]
    keys = v["deliver_keys"]
    deliverables = {
        k: {
            "plan_yaml": _plan_yaml(m, k),
            "loader": {"session_id": m["session_id"], "file": f"loader_{_key_file(k)}.pkl"},  # sid 相对寻址
            "terminal_utility": v["models"][k]["u_full"],
            "findings": v["models"][k]["findings"],
        }
        for k in keys
    }
    return {
        "session_id": m["session_id"],
        "accepted": True,  # 恒真——test 读即终局，无拒收重试
        "terminal_utility": {k: v["models"][k]["u_full"] for k in keys},
        "rank_consistency": v["rank"],
        "notes": v["notes"],
        "deliverables": deliverables,
        "report": {"board": _board(m), "steps": m["steps"]},
        "decision_log": m["decision_log"],
    }


def _plan_yaml(m: dict, key: str) -> str:
    """per-model 可独立重放计划：load → x_y → split → [FE 节点] → 族节点（plan_params
    = pin 常量 + θ* 全 pin + 资源轴 full + n_trials=0——确定性重放，不重搜索）。"""
    import inspect

    entry = m["models"][key]
    fams = registry.families()
    pyfunc = fams[entry["family"]]["pyfunc"]

    steps = [
        "- load__file:",
        "    name: 导入冻结数据",
        f"    params:\n      file_path: {_json(m['source_file'])}",
        "- x_y:",
        "    name: 划分目标",
        f"    params:\n      targets: [{_json(m['target'])}]\n      task: {_json(m['task'])}",
        "- split:",
        "    name: CV 划分",
        f"    params:\n      cv: {m['protocol'].get('cv', 5)}",
    ]
    if entry["fe"]["cat"] in registry.FE_CAT_NODES:
        steps += [f"- x_transformer__{registry.FE_CAT_NODES[entry['fe']['cat']]}:", "    name: 类别编码"]
    if entry["fe"]["num"] in registry.FE_NUM_NODES:
        steps += [f"- x_transformer__{registry.FE_NUM_NODES[entry['fe']['num']]}:", "    name: 数值缩放"]

    params = dict(entry["plan_params"])
    params.pop("n_trials", None)
    params.pop("scoring", None)
    if "n_trials" in inspect.signature(pyfunc).parameters:
        params["n_trials"] = 0  # 不重搜索——plan_params 已全 pin，重放即终训
    lines = [f"- model__{entry['family']}:", f"    name: 终局训练-{key}"]
    if params:
        lines.append("    params:")
        lines += [f"      {k}: {_json(v)}" for k, v in params.items()]
    return "\n".join(steps + lines)


# ── 候选生成：模型在前、FE 在后 ──


def propose_actions(session_id: str) -> list[dict]:
    """候选生成（菜单次序 = 决策次序：模型候选在前，FE 候选在其后）。

    - 未跑族：run 提案 + 空间摘要（动态维/静态常量——签名即声明）+ 资源轴标注
    - 已跑族：内源出题——折间 std 高 → 加深搜索（n_trials↑）；重跑合法（新 FE
      组合下重跑 = 新板面条目）
    - FE 候选（模型已选定后出现）：对 leader 非冗余槽位 × 词表内未设置档位
    - 终态（收局/弃局）返回 []
    """
    m = _store().read_manifest(session_id)
    if _is_terminal(m):
        return []
    fams = registry.families()
    proposals: list[dict] = []
    for fam in _candidates(m):
        if fam not in {v["family"] for v in m["models"].values()}:
            proposals.append(
                {
                    "action": fam,
                    "params": {},
                    "rationale": "空间摘要见 evidence（含资源参数默认档）——pin/收窄自由参数化",
                    "evidence": {"space": _space_summary(fams[fam]["pyfunc"])},
                }
            )
        else:
            entry = next(v for v in m["models"].values() if v["family"] == fam)
            deeper = max(8, 3 * int(entry["n_trials"] or 1))
            std_note = f"折间 std={entry['std']}" if entry["std"] is not None else "无搜索（固定超参）"
            proposals.append(
                {
                    "action": fam,
                    "params": {"n_trials": deeper},
                    "rationale": f"重跑加深搜索（{std_note}；u_inner={entry['u_inner']}）——SH 赌注的资源再投入",
                    "evidence": {"u_inner": entry["u_inner"], "std": entry["std"], "n_trials": entry["n_trials"]},
                }
            )

    leader = _leader(m)
    if leader is not None:
        st = fams[leader["family"]]["mdp_family"]["structure"]
        fe = m["pipeline"]["fe"]
        for slot, nodes in (("cat", registry.FE_CAT_NODES), ("num", registry.FE_NUM_NODES)):
            if slot == "cat" and (st["cat_support"] == "native" or not _has_cat(m)):
                continue  # 命中规则 ②：leader 原生类别 / 数据无类别列——cat 槽不出现
            if slot == "num" and not st["scaling_need"]:
                continue  # 命中规则 ②：leader 不需缩放——num 槽不出现
            for option, node in sorted(nodes.items()):
                if option == fe[slot]:
                    continue
                proposals.append(
                    {
                        "action": node,
                        "params": {},
                        "rationale": f"leader {leader['family']} 的 {slot} 槽 {fe[slot]} → {option}（换 FE 后重跑出新板面条目）",
                        "evidence": {"slot": slot, "current": fe[slot], "leader": leader["family"], "structure": st},
                    }
                )
    return proposals


def _space_summary(fn) -> dict:
    """签名空间摘要：动态维（Space 声明，含边界）vs 静态常量——超参面对 agent
    浮出（黑盒开箱）；plumbing 参数不入摘要。"""
    import inspect

    from mflowy.model.model.study import ContinuousSpace, DiscreteSpace

    skip = {"train_split", "x_preprocessors", "scoring", "n_trials", "evaluate_holdout", "resource"}
    dynamic, static = {}, {}
    for name, p in inspect.signature(fn).parameters.items():
        if name in skip or p.default is inspect.Signature.empty:
            continue
        if isinstance(p.default, ContinuousSpace):
            dynamic[name] = {"lo": p.default.start, "hi": p.default.end, "step": p.default.step}
        elif isinstance(p.default, DiscreteSpace):
            dynamic[name] = list(p.default)
        else:
            static[name] = p.default
    return {"dynamic": dynamic, "static": static}


# ── 观测 ──


def _observe(m: dict, reward: float | None = None) -> dict:
    """观测响应（#29 Observation 契约下限 + 域加字段）：reward = u_inner。"""
    board = _board(m)
    viable = sorted(((v["u_inner"], k) for k, v in board.items() if v["u_inner"] is not None), key=lambda t: -t[0])
    return {
        "session_id": m["session_id"],
        "candidates": _candidates(m),
        "pipeline": {"fe": dict(m["pipeline"]["fe"])},
        "board": board,
        "best": {"structure": viable[0][1], "u_inner": viable[0][0]} if viable else None,
        "budget": m["max_steps"] - m["steps"],
        "steps": m["steps"],
        "max_steps": m["max_steps"],
        "should_stop": bool(m.get("finished")) or m["steps"] >= m["max_steps"],
        "state_digest": _digest(m),
        "reward": reward,
        "decision_log": m["decision_log"],
    }


def _json(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def save_loader(session_id: str, loader, key: str) -> None:
    """run 后持久化 loader 门面（键 = 结构键；多模型会话各持一份，sid 相对寻址）。"""
    with open(_store().session_dir(session_id) / f"loader_{_key_file(key)}.pkl", "wb") as f:
        pickle.dump(loader, f)


def load_loader(session_id: str, key: str):
    with open(_store().session_dir(session_id) / f"loader_{_key_file(key)}.pkl", "rb") as f:
        return pickle.load(f)
