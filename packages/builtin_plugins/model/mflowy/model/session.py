"""建模会话引擎（应用层）：#62 回路与状态机——s 元组（光标 + 账本）+ 八动作面。

状态 ``s = (data_fingerprint, 光标 m/H/h_space/seed/F, 账本 board, oof, budget,
done, history)``；动作面 ``A = {select_model, set_hp, set_hp_space, set_seed,
select_FE}（配置编辑：零步费零奖励）∪ {run_hpo, final_train_test}（计算动作：吃步）
∪ {done}``（= finish_session 控制面独立入口，同 clean 域）。

- **run_hpo**：sqlite study per (family,F) 增量续跑（load_if_exists）；增量判据 =
  目标同一性（空间收窄/pin/加深继续，seed/scoring 变重开）；搜索结束在 θ* 上出
  OOF 预测（折外）挂 run（run_id 寻址）；reward = Δu_inner（新键 = u_inner）
- **final_train_test**：先搜后门店——θ* 从账本读 → 全量拟合 → holdout 终评恰一次
  → 产物挂 run → combo 冻结（同键不可再 run_hpo——掐断用 test 反复挑参）；reward = 0
- **账本真源** = 盘上 sqlite study（append-only）；账本条目记指针
  {study 文件, n_trials_at_last_eval}，u_inner/θ* 派生自 study；rollback 只回退光标
  ——board 永不与盘分叉（#58 P1 根治）
- **propose**：pull-only 单源谓词机器（类型可用(s) ∧ 参数合法(s,a)）——读 s 字段
  不读上一动作；词表外动作由 apply 边界 fail-loud 拒绝（带候选清单）
- split.pkl 归会话层（sid 寻址，与 study/loader 同工件命名空间）；mlflow 拿血绘认证
  不拿工件——run tag 补划分身份（mflowy.split_fingerprint + test_size/group_by）
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import uuid
from pathlib import Path

import pandas as pd
from mflowy.driver.mdp_session import SessionStore
from mflowy.model import registry
from mflowy.model.model._pipeline import oof_predictions, u_inner_of
from mflowy.model.model._split import materialize_split
from mflowy.model.model.types import MetricName
from mflowy.model.model.utils import MACHINERY_SKIP, infer_metric, normalize_space, space_from_json, space_to_json

logger = logging.getLogger(__name__)

# 八动作面（done = finish_session 控制面独立入口）；compute = 吃步
ACTIONS = ("select_model", "set_hp", "set_hp_space", "set_seed", "select_FE", "run_hpo", "final_train_test")
COMPUTE_ACTIONS = frozenset({"run_hpo", "final_train_test"})
# 配置编辑词表（node 值 = 动作面名；族名/FE 节点名不再直接 apply）
DEFAULT_N_TRIALS = 10


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
    from dataclasses import replace

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
    """候选面（族级）：卡 candidates（manifest 结构化清单）∩ 已声明族目录。"""
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


# 空间声明/持久化的单源件住 model/utils（与签名空间面同居）：normalize_space /
# space_to_json / space_from_json / MACHINERY_SKIP——session 不留第二份拷贝


# ── 开局：协议谈判在剧本层（model-brief / research-review）完成，ProtocolCard 传入即物化冻结 ──


def start_session(
    file_path: str, card: dict | None = None, cv_card: dict | None = None, max_steps: int | None = None
) -> dict:
    """开局建模会话（cv_card 必填，唯一主通道）：D 物化冻结 + 光标就位。

    协议谈判在剧本层完成——episode = (冻结 D, 冻结协议, 单一预算)，改协议 =
    新 episode（弃局重开）。
    """
    card = card or {}
    if not cv_card:
        raise ValueError("cv_card 必填——协议谈判在剧本层（model-brief）完成，引擎不再提供无协议开局")
    df = pd.read_parquet(file_path) if file_path.endswith(".parquet") else pd.read_csv(file_path)
    target = card.get("target") or cv_card.get("target")

    # 四类相容校验门：task↔dtype/metric↔task/cv↔数据/预算与候选（含 hints ⊆ 签名空间、baseline 恰一）
    from mflowy.model.protocol import CVCard

    proto = CVCard.build(
        df,
        target=target,
        **{k: v for k, v in cv_card.items() if k not in ("target", "exploratory")},
    ).as_dict()

    session_id = f"ms_{uuid.uuid4().hex[:10]}"
    store = _store()
    d = store.root / session_id
    # 不走 store.create()：manifest 落盘是 commit 点，须在 split.pkl 之后
    d.mkdir(parents=True, exist_ok=False)

    cursor = {"m": None, "H": {}, "h_space": {}, "seed": None, "F": {"cat": "passthrough", "num": "passthrough"}}
    manifest = {
        "session_id": session_id,
        "source_file": str(file_path),
        "target": target,
        "card": card,
        "protocol": {},
        "task": None,
        "id_columns": list(card.get("id_columns") or ()),
        "split": {},  # 划分身份（run tag 面板）：fingerprint + test_size + group_by
        "candidates": list(proto.get("candidates") or []),  # #63 结构化候选（含 baseline/hints/seed）
        "cursor": cursor,
        # 光标快照：rollback 弹栈只回退光标——账本/study 永不回退（盘上 append-only）
        "snapshots": [{"m": None, "H": {}, "h_space": {}, "seed": None, "F": dict(cursor["F"])}],
        "models": {},  # 账本：per (family,F) 条目（真源 = 盘上 sqlite study）
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
    """D 物化冻结（x_y + split + split.pkl——会话层决策，DAG 路径不落盘）。"""
    from mflowy.data.types import TASKTYPE as _TT
    from mflowy.driver.config import parse_enum

    target = m["target"]
    raw_task = str(protocol.get("task", "")).strip()
    if raw_task.upper() in _TT.__members__:
        raw_task = raw_task.upper()
    task = parse_enum(_TT, raw_task) if raw_task else None

    x_y_fn = _get_node("x_y")
    xy = x_y_fn(load=df, targets=target, task=task)
    split_fn = _get_node("split")
    spec = split_fn(x_y=xy, cv=protocol.get("cv", 5))
    ts = materialize_split(spec, xy)
    with open(_store().session_dir(m["session_id"]) / "split.pkl", "wb") as f:
        pickle.dump(ts, f)
    m["task"] = task.value
    m["protocol"] = protocol
    m["split"] = {
        "fingerprint": spec.data_fingerprint,
        "test_size": getattr(spec.outer, "test_size", None),
        "group_by": spec.group_by,
    }


def _get_node(name: str):
    from mflowy.driver.discover import get_node

    return get_node(name, raw=True)


# ── apply：八动作面分派（clean 形状） ───────────────────────────────────────


def apply_action(session_id: str, node: str, params: dict | None = None) -> dict:
    """执行一个回合动作：node ∈ 八动作面词表（配置编辑 + 计算动作；done = finish）。

    配置编辑只改光标（零步费零奖励，生效于该键下一次 run_hpo）；计算动作吃步。
    词表外（族名/FE 节点名直 apply 已随 #62 退役）fail-loud 带词表清单。
    """
    params = dict(params or {})
    store = _store()
    m = store.read_manifest(session_id)
    if _is_terminal(m):
        raise RuntimeError("会话已终态（收局/弃局）：动作面为空")

    if node not in ACTIONS:
        raise ValueError(
            f"动作 `{node}` 不在词表（八动作面 = {list(ACTIONS)}；done = finish 控制面独立入口，"
            "rollback/finish/abandon 同为独立入口）"
        )
    dispatch = {
        "select_model": _select_model,
        "set_hp": _set_hp,
        "set_hp_space": _set_hp_space,
        "set_seed": _set_seed,
        "select_FE": _select_fe,
        "run_hpo": _run_hpo,
        "final_train_test": _final_train_test,
    }
    return dispatch[node](store, m, params)


def _check_budget(m: dict) -> None:
    if m["steps"] >= m["max_steps"]:
        raise RuntimeError(
            f"预算耗尽（max_steps={m['max_steps']}，预算 = 实验数：只有计算动作吃步）：请 finish 或 abandon"
        )


def _require_cursor_model(m: dict) -> str:
    fam = m["cursor"]["m"]
    if not fam:
        raise RuntimeError("先 select_model 选定候选——配置编辑与计算动作都作用于当前候选配置")
    return fam


def _push_snapshot(m: dict) -> None:
    c = m["cursor"]
    m["snapshots"].append(
        {"m": c["m"], "H": dict(c["H"]), "h_space": dict(c["h_space"]), "seed": c["seed"], "F": dict(c["F"])}
    )


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


def _commit(store, m: dict, action: str, params: dict, note: str = "", reward: float | None = None) -> dict:
    _log(m, action, params, note=note)
    store.write_manifest(m["session_id"], m)
    return _observe(m, reward=reward)


# ── 配置编辑动作（零步费零奖励） ────────────────────────────────────────────


def _select_model(store, m: dict, params: dict) -> dict:
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
        # 切族：清 pin/收窄后注入该候选的 hints 缺省（卡是先验不是边界——π 回合内可覆盖）
        cand = _candidate_of(m, family)
        hints = cand.get("space_hints") or {}
        m["cursor"]["H"] = {k: v for k, v in hints.items() if not isinstance(v, dict)}
        m["cursor"]["h_space"] = {k: v for k, v in hints.items() if isinstance(v, dict)}
        if m["cursor"]["seed"] is None and cand.get("seed") is not None:
            m["cursor"]["seed"] = cand["seed"]  # 候选缺省 seed（未显式 set_seed 时生效）
    return _commit(store, m, "select_model", {"family": family}, note=f"光标候选 → {family}", reward=0)


def _set_hp(store, m: dict, params: dict) -> dict:
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
    _push_snapshot(m)
    m["cursor"]["H"].update(params)
    return _commit(store, m, "set_hp", dict(params), note=f"pin {sorted(params)}", reward=0)


def _set_hp_space(store, m: dict, params: dict) -> dict:
    if not params:
        raise ValueError("set_hp_space 须带至少一个 {param: [lo, hi] | [v1, v2, …]} 收窄")
    family = _require_cursor_model(m)
    face = _family_params(family)
    normalized = {}
    for param, value in params.items():
        if param not in face:
            raise ValueError(f"not_a_param：`{param}` 不在族 `{family}` 签名超参面（可用：{sorted(face)}）")
        normalized[param] = space_to_json(normalize_space(param, face[param], value))
    _push_snapshot(m)
    m["cursor"]["h_space"].update(normalized)
    return _commit(store, m, "set_hp_space", dict(params), note=f"收窄 {sorted(params)}", reward=0)


def _set_seed(store, m: dict, params: dict) -> dict:
    unknown = set(params) - {"seed"}
    if unknown:
        raise ValueError(f"set_seed 只收 seed，多余参数 {sorted(unknown)}")
    seed = params.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed 须为 int（搜索的环境参数——换 seed = 换目标 = study 重开），got {seed!r}")
    _push_snapshot(m)
    m["cursor"]["seed"] = seed
    return _commit(
        store, m, "set_seed", {"seed": seed}, note=f"seed={seed}（下次 run_hpo 生效；目标变即 study 重开）", reward=0
    )


def _select_fe(store, m: dict, params: dict) -> dict:
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
    return _commit(
        store,
        m,
        "select_FE",
        {"node": node},
        note=f"{slot} 槽 {cur} → {option}（换 FE 后 run_hpo 出新账本条目）",
        reward=0,
    )


# ── 计算动作：run_hpo（增量搜索 + OOF 观察面） ─────────────────────────────


def _machinery_of(family: str):
    meta = registry.families()[family]["mdp_family"]
    if "machinery" not in meta:
        raise ValueError(f"族 `{family}` 未声明 machinery（.mdp_family['machinery']——两行为共享的 estimator 机器）")
    return meta["machinery"], meta.get("fit_params")


def _effective_overrides(m: dict) -> dict:
    """光标有效参数覆盖：h_space（Space 收窄）∪ H（pin，pin 优先）。"""
    out = {k: space_from_json(v) for k, v in m["cursor"]["h_space"].items()}
    out.update(m["cursor"]["H"])
    return out


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


def _open_study(m: dict, key: str, objective_id: dict, scoring: MetricName):
    """盘上 study 真源：目标同一性判据——空间变（收窄/pin/加深）增量继续，目标变（seed/scoring）重开。"""
    import optuna

    db = _store().session_dir(m["session_id"]) / f"study_{_key_file(key)}.db"
    storage = f"sqlite:///{db}"
    direction = "maximize" if MetricName.higher_is_better(scoring) else "minimize"
    if db.exists():
        study = optuna.load_study(study_name=key, storage=storage)
        if study.user_attrs.get("objective") == objective_id:
            return study
        db.unlink()  # 目标变 → 重开（trial 清零；旧文件弃置——append-only 真源换新目标）
    study = optuna.create_study(study_name=key, storage=storage, direction=direction, load_if_exists=True)
    study.set_user_attr("objective", objective_id)
    return study


def _tag_run(run, m: dict, *, action: str, key: str, family: str) -> None:
    """mlflow 血缘认证（不拿工件）：划分身份 + 会话/组合身份。"""
    import mlflow

    mlflow.set_tags(
        {
            "mflowy.session_id": m["session_id"],
            "mflowy.action": action,
            "mflowy.combo_key": key,
            "mflowy.family": family,
            "mflowy.split_fingerprint": m["split"]["fingerprint"],
            "mflowy.split_test_size": str(m["split"].get("test_size")),
            "mflowy.split_group_by": str(m["split"].get("group_by")),
        }
    )


def _run_hpo(store, m: dict, params: dict) -> dict:
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
    _check_budget(m)
    fe = _ensure_structure(m, family)
    key = _structure_key(family, fe)
    entry = m["models"].setdefault(key, _new_entry(key, family, fe))
    if entry["frozen"]:
        raise RuntimeError(f"combo `{key}` 已冻结（final_train_test 后同键不可再搜索——掐断用 test 反复挑参）")

    scoring = _metric_of(m)
    seed = m["cursor"]["seed"]
    ts = _split_for_run(m)
    ct = registry.build_ct(fe["cat"], fe["num"], (ts.X_train, ts.y_train, _task(m)))
    machinery, _ = _machinery_of(family)
    make_pipe, base, space = machinery(_task(m), ct, seed=seed, **_effective_overrides(m))
    if not space and n_trials > 1:
        raise ValueError("全 pin 候选（有效空间为空）走 n_trials=1 退化形——多余 trial 是同一参数的重复评估")

    objective_id = {"seed": seed if seed is not None else "default", "scoring": scoring.value}
    study = _open_study(m, key, objective_id, scoring)
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
            runs_dir = _store().session_dir(m["session_id"]) / "runs" / run.info.run_id
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
            "best_params": dict(best.params),  # θ*（动态维；pin 常量在光标 H）
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
    m["steps"] += 1
    resp = _commit(
        store,
        m,
        "run_hpo",
        {"n_trials": n_trials, "key": key},
        note=f"u_inner={new_u}（Δ={reward}）trials={len(study.trials)} oof_run={run.info.run_id[:8]}",
        reward=reward,
    )
    resp["findings"] = findings  # run 响应浮出（π 的过程决策面）
    return resp


def _final_train_test(store, m: dict, params: dict) -> dict:
    """final_train_test：先搜后门店——θ* 从账本读 → 全量拟合 → holdout 终评恰一次 → 冻结。"""
    import mlflow
    from mflowy.model.model._pipeline import holdout_report, training

    if params:
        raise ValueError("final_train_test 无参数——θ* 恒从账本读（先搜后店：无搜索直训不合法）")
    family = _require_cursor_model(m)
    _check_budget(m)
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

    scoring = _metric_of(m)
    seed = entry.get("objective", {}).get("seed")  # 搜索环境种子（θ* 的身份分量）
    ts = _split_for_run(m)
    ct = registry.build_ct(fe["cat"], fe["num"], (ts.X_train, ts.y_train, _task(m)))
    machinery, fit_params_of = _machinery_of(family)
    overrides = {**_effective_overrides(m), **entry["best_params"]}
    make_pipe, base, space = machinery(_task(m), ct, seed=seed, **overrides)
    if space:
        raise RuntimeError(
            f"θ* + pin 后有效空间非空（{sorted(space)}）——best_params 须覆盖全部动态维；"
            "账本/光标不一致，疑似缺陷，请携带 session_id 反馈"
        )
    fitp = fit_params_of() if fit_params_of else {}

    with mlflow.start_run(run_name=f"final_train_test:{key}") as run:
        _tag_run(run, m, action="final_train_test", key=key, family=family)
        from mflowy.utils.capture import capture_prints

        # 通道纪律（M7）：同 run_hpo——直调 _pipeline 的自建捕获窗口
        with capture_prints() as buf:
            loader = training(ts, family, make_pipe, base, fit_params=fitp, scoring=scoring)
            report = holdout_report(loader, ts, scoring)  # verdict 面（u_full/findings/长表）
            runs_dir = _store().session_dir(m["session_id"]) / "runs" / run.info.run_id
            runs_dir.mkdir(parents=True, exist_ok=True)
            y_pred_df = report["y_pred"]
            if not isinstance(y_pred_df, pd.DataFrame):
                y_pred_df = pd.DataFrame(y_pred_df, index=ts.y_test.index, columns=ts.y_test.columns)
            y_pred_df.to_parquet(runs_dir / "holdout_y_pred.parquet", index=True)
            save_loader(m["session_id"], loader, key)
        captured = buf.getvalue().strip()
        if captured:
            logger.info("final_train_test[%s] 捕获输出：\n%s", key, captured)

    entry.update(
        {
            "frozen": True,
            "u_full": report["u_full"],
            "metrics": {t: {k.value: v for k, v in tm.items()} for t, tm in report["metrics"].items()},
            "findings": [{"source": "rule", **f} for f in report["findings"]]
            + [f for f in entry.get("findings", []) if f.get("source") == "llm"],
            "run_id": run.info.run_id,
            "final_params": {k: v for k, v in base.items() if k in _family_params(family)},  # plan 面（签名内）
        }
    )
    m["steps"] += 1
    return _commit(
        store,
        m,
        "final_train_test",
        {"key": key},
        note=f"u_full={report['u_full']} combo 冻结 run={run.info.run_id[:8]}",
        reward=0,
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
        "std": None,
        "frozen": False,
        "u_full": None,
        "metrics": None,
        "findings": [],
        "run_id": None,
        "final_params": None,
        "oof": None,
    }


# ── 控制面 ──


def report_findings(session_id: str, key: str, findings: list[dict]) -> dict:
    """LLM findings 落诊（#36 控制面）：不可机械化诊断进 models[key]——schema 校验、
    决策日志审计、与规则 findings 均匀聚合（propose 消费面不区分来源）。

    schema = diagnose 族 ErrorReport 先例（kind/severity/evidence/message）+
    ``source`` 引擎保留键；kind ⊆ 已知词表（#35 定案——语言超纲即拒）。
    """
    store = _store()
    m = store.read_manifest(session_id)
    if _is_terminal(m):
        raise RuntimeError("会话已终态（收局/弃局）：落诊面已关闭")
    if key not in m["models"]:
        raise ValueError(f"落诊键 `{key}` 不在账本（已训 combo：{sorted(m['models'])}）")
    validated = [_validate_llm_finding(f) for f in findings]

    entry = m["models"][key]
    entry["findings"] = [f for f in entry.get("findings", []) if f.get("source") == "rule"] + validated
    _log(m, "report_findings", {"key": key, "n": len(validated)}, note="; ".join(f["kind"] for f in validated))
    store.write_manifest(session_id, m)
    resp = _observe(m)
    resp["findings"] = validated
    return resp


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


def rollback(session_id: str) -> dict:
    """回退光标到上一快照（决策日志保留；预算不回补；账本/study 永不回退——盘上 append-only）。"""
    store = _store()
    m = store.read_manifest(session_id)
    if _is_terminal(m):
        raise RuntimeError("会话已终态（收局/弃局）：不可回滚——决策日志与模型账本是审计面")
    if len(m["snapshots"]) <= 1:
        raise RuntimeError("无历史版本可回退")
    snap = m["snapshots"].pop()
    m["cursor"] = {
        "m": snap["m"],
        "H": dict(snap["H"]),
        "h_space": dict(snap["h_space"]),
        "seed": snap["seed"],
        "F": dict(snap["F"]),
    }
    _log(m, "rollback", {}, note=f"光标弹栈（预算不回补：steps={m['steps']}/{m['max_steps']}；账本与盘上 study 一致）")
    store.write_manifest(session_id, m)
    return _observe(m)


def abandon_session(session_id: str, reason: str | None = None) -> dict:
    """弃局终局：委托 #29 SessionStore.abandon——置 abandoned、reason 落盘（审计）、
    保留决策日志与模型账本、不导出交付物；已终态再弃即 raise。"""
    return _store().abandon(session_id, reason=reason)


# ── finish（done）：一次结算、多模型交付 ────────────────────────────────────


def finish_session(session_id: str, *, pdf: str | None = None) -> dict:
    """收局（done，#64 两段式——报告是 π 的产物，引擎只收编不编译）：

    - ``pdf=None``（preview）：聚合全部已训 combo 的产物清单（OOF 图集/预测长表/
      指标）+ 绘泰勒 + decision_log/board 素材——π（LLM 或人）用宿主多模态读取
      后撰写终局报告；零副作用（不落 verdict，manifest 不动），可重入
    - ``pdf=...``（真收局）：收编 π 产报告进 deliverables（复制 + tmp/rename 原子
      落盘；文件缺失 fail-loud）+ verdict（u_full/findings/rank 一致性）。收编不验
      内容——报告是给人看的便当，数字权威面是 verdict/board/盘上 study；重入 =
      幂等重导出（交付报告不可变更——重写须弃局重开）

    accepted 恒真（holdout 读在 final_train_test 恰一次，finish 只聚合）。
    """
    store = _store()
    m = store.read_manifest(session_id)
    if m.get("abandoned"):
        raise RuntimeError("会话已弃局：不可收局（弃局不可复活，改协议重开走新会话）")

    if pdf is None:
        if _is_terminal(m):
            raise RuntimeError("会话已终态（收局/弃局）：preview 面已关闭")
        return _finish_preview(m)

    if m.get("verdict"):  # 已结算：幂等重导出（交付报告不可变更——审计面）
        adopted = m["verdict"]["report"]["source_pdf"]
        if pdf != adopted:
            raise ValueError(f"已收局交付报告不可变更（审计面）——重写报告须弃局重开；已收编报告源：{adopted!r}")
        return _verdict_response(m)

    keys = sorted(k for k, v in m["models"].items() if v["frozen"])
    if not keys:
        raise RuntimeError("无冻结 combo 可交付——先 run_hpo + final_train_test 再收局")

    src = Path(pdf)
    if not src.is_file():
        raise ValueError(f"报告文件不存在：{pdf!r}（先 pdf=None 拿素材看图，π 产报告后带路径收局）")
    if len(keys) >= 2:
        taylor_report(session_id, [m["models"][k]["run_id"] for k in keys])  # 终局泰勒图（收编素材）
    report = _adopt_report(store.session_dir(session_id), src)

    verdict_models = {k: {"u_full": m["models"][k]["u_full"], "findings": m["models"][k]["findings"]} for k in keys}
    m["finished"] = True
    m["verdict"] = {
        "deliver_keys": keys,
        "models": verdict_models,
        "rank": _rank_consistency(m, keys),
        "report": report,
    }
    _log(m, "finish", {"pdf": str(src)}, note=f"交付 {keys}；报告 {report['file']}")
    store.write_manifest(session_id, m)
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


def _finish_preview(m: dict) -> dict:
    """preview 面：产物清单聚合 + 泰勒——只读（除泰勒图落盘，manifest 零 mutation）。"""
    artifacts = []
    frozen_keys = []
    for key, e in sorted(m["models"].items()):
        if e.get("u_inner") is None:
            continue  # 未搜 combo 无观察面
        art = {
            "key": key,
            "family": e["family"],
            "oof_y_pred": e["oof"]["y_pred"],
            "plots": [f"runs/{e['oof']['run_id']}/plots/{p}" for p in e["oof"].get("plots", [])],
            "u_inner": e["u_inner"],
            "findings": e["findings"],
        }
        if e["frozen"]:
            frozen_keys.append(key)
            art["holdout_y_pred"] = f"runs/{e['run_id']}/holdout_y_pred.parquet"
            art["u_full"] = e["u_full"]
            art["metrics"] = e["metrics"]
        artifacts.append(art)
    taylor = (
        taylor_report(m["session_id"], [m["models"][k]["run_id"] for k in frozen_keys])
        if len(frozen_keys) >= 2
        else None
    )
    return {
        "session_id": m["session_id"],
        "preview": True,
        "artifacts": artifacts,
        "taylor": str(taylor["figure"]) if taylor and taylor["figure"] else None,
        "taylor_stats": taylor["stats"] if taylor else {},
        "board": _board(m),
        "steps": m["steps"],
        "max_steps": m["max_steps"],
        "decision_log": m["decision_log"],
    }


def taylor_report(session_id: str, run_ids: list[str]) -> dict:
    """泰勒聚合门面（工具与 finish preview 共用）：stats 键 = combo key（run_id 反查
    账本——agent 可读面）；figure 为会话目录相对路径。"""
    from mflowy.model.report import taylor_for_runs

    m = _store().read_manifest(session_id)
    out = taylor_for_runs(_store().session_dir(session_id), run_ids, m["task"])
    rid_to_key = {v["run_id"]: k for k, v in m["models"].items() if v.get("run_id")}
    out["stats"] = {rid_to_key.get(rid, rid): row for rid, row in out["stats"].items()}
    return {**out, "figure": out["figure"] if out["figure"] is None else str(out["figure"])}


def _rank_consistency(m: dict, keys: list[str]) -> float | None:
    """SH 赌注出账：交付集上 u_inner 与 u_full 的秩相关。样本 < 2 返回 None——不伪装。"""
    import scipy.stats as st

    pairs = [(m["models"][k]["u_inner"], m["models"][k]["u_full"]) for k in keys]
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
            "run_id": m["models"][k]["run_id"],
            "terminal_utility": v["models"][k]["u_full"],
            "findings": v["models"][k]["findings"],
        }
        for k in keys
    }
    return {
        "session_id": m["session_id"],
        "accepted": True,  # 恒真——holdout 读即终局，无拒收重试
        "terminal_utility": {k: v["models"][k]["u_full"] for k in keys},
        "rank_consistency": v["rank"],
        "report": v.get("report"),
        "deliverables": deliverables,
        "board": _board(m),
        "steps": m["steps"],
        "decision_log": m["decision_log"],
    }


def _plan_yaml(m: dict, key: str) -> str:
    """per-combo 可独立重放计划：load → x_y → split → [FE 节点] → 族节点（final_params
    全 pin → 空间空 → training 行为，确定性重放）。"""
    entry = m["models"][key]
    fe = entry["fe"]
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
    if fe["cat"] in registry.FE_CAT_NODES:
        steps += [f"- x_transformer__{registry.FE_CAT_NODES[fe['cat']]}:", "    name: 类别编码"]
    if fe["num"] in registry.FE_NUM_NODES:
        steps += [f"- x_transformer__{registry.FE_NUM_NODES[fe['num']]}:", "    name: 数值缩放"]
    params = dict(entry["final_params"] or {})
    lines = [f"- model__{entry['family']}:", f"    name: 终局训练-{key}"]
    if params:
        lines.append("    params:")
        lines += [f"      {k}: {_json(v)}" for k, v in params.items()]
    return "\n".join(steps + lines)


# ── 候选生成：单源谓词机器（类型可用(s) ∧ 参数合法(s,a)） ───────────────────


def propose_actions(session_id: str) -> list[dict]:
    """菜单（pull-only，#38 规则引擎基座）：findings 驱动层（fire + 记分牌序，浮于
    顶部）+ 基座谓词机层（类型可用(s) ∧ 参数合法(s,a)——策展序）。

    - findings 驱动：fold_std → 加深（n_trials 门控回归防线）；complexity_edge /
      overfit_gap → 正则收窄/pin（族自声明 .regularizers）；slice_collapse →
      类别权重提示；error_concentrated → FE 换档（规则挂词表节点——知识随词汇
      主人居住）；无 leader（S 无模型信息）时 FE 候选零出现（引擎级 gate）
    - 基座层：select_model 候选（baseline 第一、未跑在前、已训未全冻结降级末位）、
      run_hpo 首搜、final_train_test 交付、配置编辑动作面
    - LLM findings（#36）与规则 findings 均匀聚合（同一 fire 入口，无来源区分）
    - 终态（收局/弃局）返回 []
    """
    import logging

    from mflowy.driver.discover import get_node
    from mflowy.driver.rule import fire, prioritize
    from mflowy.model import propose_rules

    m = _store().read_manifest(session_id)
    if _is_terminal(m):
        return []
    propose_rules.attach_rules()  # 幂等：族级规则（.regularizers 面）+ FE 词表规则

    fams = registry.families()
    cursor = m["cursor"]
    key = _structure_key(cursor["m"], cursor["F"]) if cursor["m"] else None
    entry = m["models"].get(key) if key else None

    # ── findings 驱动层 ──
    findings = [{"key": k, **f} for k, e in m["models"].items() for f in (e.get("findings") or [])]
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
    nodes = [fams[f]["pyfunc"] for f in fams] + [
        get_node(f"x_transformer__{node}", raw=True) for node in registry.fe_vocab()
    ]
    result = fire(findings, ctx, nodes)
    if result.unconsumed:
        logging.getLogger(__name__).warning("findings 无规则订阅（无静默丢弃——补订阅或修 kind）：%s", result.unconsumed)
    driven = prioritize(result.candidates)

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
    # 已训未全冻结：降级末位（θ*/u_inner/u_full 并排）
    for fam in ordered:
        entries = [v for v in m["models"].values() if v["family"] == fam]
        if entries and not all(v["frozen"] for v in entries):
            best = max(entries, key=lambda v: v["u_inner"] if v["u_inner"] is not None else float("-inf"))
            proposals.append(
                {
                    "action": "select_model",
                    "params": {"family": fam},
                    "rationale": f"已训族（u_inner={best['u_inner']}）——重选可加深搜索/换 FE 组合",
                    "evidence": {
                        "tried": True,
                        "u_inner": best["u_inner"],
                        "best_params": best["best_params"],
                        "u_full": best["u_full"],
                        "frozen_combos": sum(1 for v in entries if v["frozen"]),
                    },
                }
            )

    if cursor["m"]:
        # run_hpo 首搜（已搜键的加深题走 findings 驱动层——fold_std 门控）
        if entry is None:
            proposals.append(
                {
                    "action": "run_hpo",
                    "params": {"n_trials": DEFAULT_N_TRIALS},
                    "rationale": "首搜（trial 落盘 sqlite study 增量续跑）",
                    "evidence": {"key": key, "space": _effective_space_summary(m), "n_trials": DEFAULT_N_TRIALS},
                }
            )
        elif entry.get("u_inner") is not None and not entry["frozen"]:
            proposals.append(
                {
                    "action": "final_train_test",
                    "params": {},
                    "rationale": f"终训交付（θ* 已在账本：{entry['best_params']}）——holdout 终评恰一次后 combo 冻结",
                    "evidence": {"key": key, "u_inner": entry["u_inner"], "n_trials": entry["n_trials_at_last_eval"]},
                }
            )
        proposals.append(
            {
                "action": "set_hp",
                "params": {},
                "rationale": "pin 常量（生效于下一次 run_hpo；静态常量可 pin、动态维可收窄）",
                "evidence": {"face": _pin_face(m)},
            }
        )
        proposals.append(
            {
                "action": "set_hp_space",
                "params": {},
                "rationale": "收窄动态维（空间变增量继续；seed/scoring 变 study 重开）",
                "evidence": {"face": _space_face(m)},
            }
        )
        proposals.append(
            {
                "action": "set_seed",
                "params": {},
                "rationale": "设随机种子（搜索的环境参数——换 seed = 换目标 = study 重开）",
                "evidence": {"current": cursor["seed"]},
            }
        )
        # select_FE 基座档：有模型账本信息（leader）才出现——S 无模型信息时 FE
        # 候选零出现（引擎级 gate，非权重压序；动作边界只看 cursor.m——菜单是建议不是边界）
        if m["models"]:
            _fe_base_entries(m, cursor, fams, proposals)
    return _merge_menu(driven + proposals)


def _merge_menu(items: list[dict]) -> list[dict]:
    """(action, params) 键合并（#38 会话侧补位——引擎不去重）：同动作同参 = 同一
    建议的多重证据，浮出一项——rationale 并集（保序去重、"；"连接）；命中的规则
    溯源并列进 ``evidence["co_hits"]``（每条含 rationale 与 ``_rule`` 分解——合并
    不静默）。记分牌位与 ``_score`` 保首现实体（prioritize 已按分排序——排序
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
                    "action": "select_FE",
                    "params": {"node": node},
                    "rationale": f"{cursor['m']} 的 {slot} 槽 {cursor['F'][slot]} → {option}（换 FE 后 run_hpo 出新账本条目）",
                    "evidence": {"slot": slot, "current": cursor["F"][slot], "structure": st},
                }
            )


def _pin_face(m: dict) -> dict:
    """set_hp 参数面：签名超参（当前 pin 状态 + 静态/动态分型）。"""
    face = _family_params(m["cursor"]["m"])
    return {
        "params": {
            k: {"current_pin": m["cursor"]["H"].get(k), "default": _json_default(v)}
            for k, v in sorted(face.items())
            if not _is_space(v)
        }
    }


def _space_face(m: dict) -> dict:
    face = _family_params(m["cursor"]["m"])
    return {
        "params": {
            k: space_to_json(v) | {"current_narrow": m["cursor"]["h_space"].get(k)}
            for k, v in sorted(face.items())
            if _is_space(v)
        }
    }


def _effective_space_summary(m: dict) -> dict:
    """有效空间摘要：签名空间经光标收窄后的实况。"""
    face = _family_params(m["cursor"]["m"])
    out = {}
    for k, v in face.items():
        if _is_space(v):
            out[k] = m["cursor"]["h_space"].get(k) or space_to_json(v)
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


# ── 观测 ───────────────────────────────────────────────────────────────────


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
            "u_full": v["u_full"],
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


def _observe(m: dict, reward: float | None = None) -> dict:
    """观测响应（#29 Observation 契约下限 + 域加字段）：reward = Δu_inner（新键 = u_inner）。"""
    return {
        "session_id": m["session_id"],
        "candidates": _candidates(m),
        "cursor": {
            "m": m["cursor"]["m"],
            "H": dict(m["cursor"]["H"]),
            "h_space": dict(m["cursor"]["h_space"]),
            "seed": m["cursor"]["seed"],
            "F": dict(m["cursor"]["F"]),
        },
        "board": _board(m),
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
    """final_train_test 后持久化 loader 门面（键 = 结构键；多模型会话各持一份，sid 相对寻址）。"""
    with open(_store().session_dir(session_id) / f"loader_{_key_file(key)}.pkl", "wb") as f:
        pickle.dump(loader, f)


def load_loader(session_id: str, key: str):
    with open(_store().session_dir(session_id) / f"loader_{_key_file(key)}.pkl", "rb") as f:
        return pickle.load(f)
