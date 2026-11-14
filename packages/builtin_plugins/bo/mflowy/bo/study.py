"""study 状态载体——optuna sqlite db（#180 R2/#175 Q4）。

- 落位：``<task_dir>/bo_studies/<study_name>/study.db``；db 单文件 = 全部状态
  （trial 状态/值/attrs + study 身份键）。不匹配 ``*_sessions`` glob → 不被
  sessions_gc 回收（长寿命可继承资产）。
- 引擎：TPE 单引擎（#175 Q1：``multivariate=True, constant_liar=True`` 显式开；
  引擎身份 = study 确定性配置，环境不可变）。sampler 不落盘（optuna 内存态），
  每次工具调用以 ``seed + 现存 trial 数`` 重建——同一 db 内容恒出同一提案序列
  （跨机器可复现），且不重放启动期随机流（固定 seed 逐调用重建会在 startup 期
  重复同一批随机候选，#172 实测语义）。
- 身份：``search_space_fingerprint`` = hash(空间组 + direction + seed)——影响
  ``study.ask`` 结果的字段（#178 M1 同一性判据）；开局比对，不符即迁移式重建。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .types import CombinatorialDim, ContinuousDim, DiscreteDim, SpaceDim

logger = logging.getLogger(__name__)

# 开局最小历史量机器门（#183 R4）：两锚统一（parquet 行数 / db COMPLETE trial 数），
# 重复条件/平行样按行计不去重——健康地板非信息量度量；锚定 = TPE n_startup(10)×2
MIN_HISTORY_TRIALS = 20

# study 名（db 内 optuna study 名；一库一 study，路径已含 study_name）
_STUDY_NAME = "bo"

# ── study attrs 契约（全持久于 sqlite；#180 R3 + #182 R3 扩键）─────────────────
ATTR_MODEL = "model"  # {"run_id": str}——模型 = 调用入参非身份（换模型仅重置学习信号）
ATTR_TARGET = "target_column"  # direction 键目标列
ATTR_SCHEMA = "feature_schema"  # 开局湿历史列清单（ask 铺空列唯一依据，#179 R1）
ATTR_SPACE = "search_space"  # 空间组引擎面（kind/payload；编码声明不进引擎）
# 业务方向三态（maximize/minimize/match）。#180 R3「direction 用 optuna 原生
# directions 不另存」在 match 上不成立：match 与 minimize 的原生方向同为 minimize，
# 而 match-ness 决定回填 hinge 编码（bo_tell 消费）——存此键是 tell 编码的单源
ATTR_DIRECTION = "direction"
ATTR_FINGERPRINT = "search_space_fingerprint"  # 同一性哈希（开局比对）
ATTR_ENGINE = "engine"  # {"sampler": "tpe", "multivariate": true, "constant_liar": true, "seed": int}
ATTR_SIGNAL = "learning_signal"  # 现纪元信号态快照（含纪元起算 trial number）
ATTR_SIGNAL_ARCHIVE = "learning_signal_archive"  # {run_id: 信号态} 累积归档

# ── trial attrs 契约（三键判定/来源/原因 + 导出簿记键）─────────────────────────
ATTR_LABEL = "label"  # 判定标签：close | severe_mismatch | not_counted（信号唯一消费源）
ATTR_SOURCE = "source"  # 来源：recommended | spontaneous
ATTR_REASON = "reason"  # 终局原因（审计词表，与台账 attribution/叙述同源）
ATTR_MEASURED = "measured_value"  # 原始实测值（match 纪元 trial.value = 带外距离，raw 存此）
ATTR_CONDITIONS = "conditions"  # 非搜索特征列取值（钉值/实测条件——db 全量真源的面）
ATTR_ROUND = "round"  # 轮次（台账审计）
ATTR_REPLICATE_OF = "replicate_of"  # 平行样指向原推荐 trial 号

LABELS = ("close", "severe_mismatch", "not_counted")
SOURCES = ("recommended", "spontaneous")
PRUNED_REASONS = ("hard_rule_violation", "out_of_band", "user_rejected", "conditional_failure")
FAIL_REASONS = (
    "diversity_rejected",
    "abandoned",
    "operational_accident",
    "dirty_data",
    "rebuild_finalized",
    "variant_replaced",
)
# 台账 attribution（#181 R1）→ 终态 + 原因词表映射
ATTRIBUTION_MAP = {
    "conditional": ("PRUNED", "conditional_failure"),
    "operational": ("FAIL", "operational_accident"),
    "dirty_data": ("FAIL", "dirty_data"),
}


def study_root(task_dir: Path, study_name: str) -> Path:
    """study 目录：``<task_dir>/bo_studies/<study_name>/``（#180 R2）。"""
    return task_dir / "bo_studies" / study_name


def parse_search_space(raw: dict[str, Any]) -> dict[str, SpaceDim]:
    """工具入参搜索空间 → BOCard 维模型（单源校验——签名=约束=文档=校验红线）。"""
    from pydantic import TypeAdapter

    return TypeAdapter(dict[str, SpaceDim]).validate_python(raw)


def engine_space_of(dims: dict[str, SpaceDim]) -> dict[str, dict[str, Any]]:
    """空间组引擎面：排除编码声明（不影响 ask 不进同一性，#178 M1）。"""
    return {name: dim.model_dump(exclude={"encoding"}, exclude_none=True) for name, dim in dims.items()}


def fingerprint_of(dims: dict[str, SpaceDim], *, direction: str, seed: int) -> str:
    """同一性指纹：影响 study.ask 的字段（空间 kind/边界/choices + direction + seed）。"""
    payload = json.dumps(
        {"space": engine_space_of(dims), "direction": direction, "seed": seed},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def encode_objective(direction: str, raw: float, band: tuple[float, float] | None) -> float:
    """实测值 → 引擎目标值：maximize/minimize 原值；match = 到带距离（带内 0，最小化）。

    match 纪元 ``trial.value`` 为 hinge 距离、raw 恒存 ``measured_value`` attr——
    导出与台账读 attr（单一来源），band 变更（策略组不重建）只影响此后 tell 的
    编码，历史 trial 值不重算（#177 标签即事实同义）。
    """
    if direction == "match":
        if band is None:
            raise ValueError("direction=match 需要 band（lo, hi）——目标带是 match 编码的一部分")
        lo, hi = band
        return max(lo - raw, raw - hi, 0.0)
    return raw


def distributions_of(dims: dict[str, SpaceDim]) -> dict[str, Any]:
    """空间组 → optuna 分布（ask 采样与 add_trial 声明共用单源）。"""
    from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution

    out: dict[str, BaseDistribution] = {}
    for name, dim in dims.items():
        if isinstance(dim, ContinuousDim):
            if dim.dtype == "int":  # 卡校验保证 int 维边界整值（types.py）
                out[name] = IntDistribution(int(dim.low), int(dim.high), log=dim.scale == "log")
            else:
                out[name] = FloatDistribution(dim.low, dim.high, log=dim.scale == "log")
        elif isinstance(dim, DiscreteDim | CombinatorialDim):
            out[name] = CategoricalDistribution(dim.choices)
    return out


def assert_params_in_distributions(dists: dict[str, Any], params: dict[str, Any], *, where: str) -> None:
    """参数键 ⊆ 空间维且落在分布内（fail-loud 列名/值）——迁移越界与追加超范围
    共用的导入层拦截（#172：宽窄分布混用会静默整维剔除，必须在导入层显式拦截）。
    非搜索特征列不属 params（钉值/实测条件走 conditions attr，#181 R6）。"""
    unknown = sorted(set(params) - set(dists))
    if unknown:
        raise ValueError(f"{where}：参数含搜索空间外列 {unknown}——空间维：{sorted(dists)}；非搜索特征列请进 conditions")
    for name, value in params.items():
        dist = dists[name]
        ok = (
            dist.low <= value <= dist.high
            if hasattr(dist, "low")
            else (value in dist.choices if hasattr(dist, "choices") else True)
        )
        if not ok:
            raise ValueError(f"{where}：参数 '{name}'={value!r} 超出搜索分布 {dist}")


# ── study 开/建 ────────────────────────────────────────────────────────────────


def _storage(db_path: Path) -> Any:
    import optuna.storages

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.storages.RDBStorage(url=f"sqlite:///{db_path}")


def create_study(db_path: Path, *, direction: str) -> Any:
    """建新 study（不 load_if_exists——撞名即 fail-loud，#180 R1）。"""
    import optuna

    native = {"maximize": "maximize", "minimize": "minimize", "match": "minimize"}[direction]
    return optuna.create_study(study_name=_STUDY_NAME, storage=_storage(db_path), direction=native)


def open_study(db_path: Path, *, dims: dict[str, SpaceDim] | None = None) -> Any:
    """开既有 study；dims 给定时附带确定性 sampler（ask 路径）。"""
    import optuna

    if not db_path.exists():
        raise FileNotFoundError(f"study db 不存在：{db_path}")
    study = optuna.load_study(study_name=_STUDY_NAME, storage=_storage(db_path))
    if dims is not None:
        study.sampler = build_sampler(study.user_attrs[ATTR_ENGINE], len(study.get_trials(deepcopy=False)))
    return study


def assert_min_history(n: int) -> None:
    """开局最小历史量机器门（#183 R4，两锚统一：parquet 行数 / db COMPLETE 数）。"""
    if n < MIN_HISTORY_TRIALS:
        raise ValueError(
            f"开局最小历史量 {MIN_HISTORY_TRIALS}（#183 机器门：湿历史是唯一学习源，"
            f"TPE n_startup×2 保守地板；重复条件/平行样按行计不去重）——收到 {n}，差 {MIN_HISTORY_TRIALS - n}"
        )


def build_sampler(engine: dict[str, Any], n_existing_trials: int) -> Any:
    """TPE 单引擎（multivariate+constant_liar 显式开）；seed + 现存 trial 数派生
    （见模块 docstring——确定性 + 不重放启动随机流）。"""
    from optuna.samplers import TPESampler

    return TPESampler(
        multivariate=engine["multivariate"],
        constant_liar=engine["constant_liar"],
        seed=engine["seed"] + n_existing_trials,
    )


def set_identity_attrs(
    study: Any,
    *,
    model_run_id: str,
    target_column: str,
    feature_schema: list[str],
    dims: dict[str, SpaceDim],
    direction: str,
    seed: int,
) -> None:
    study.set_user_attr(ATTR_MODEL, {"run_id": model_run_id})
    study.set_user_attr(ATTR_TARGET, target_column)
    study.set_user_attr(ATTR_SCHEMA, feature_schema)
    study.set_user_attr(ATTR_DIRECTION, direction)
    study.set_user_attr(ATTR_SPACE, engine_space_of(dims))
    study.set_user_attr(ATTR_FINGERPRINT, fingerprint_of(dims, direction=direction, seed=seed))
    study.set_user_attr(
        ATTR_ENGINE,
        {"sampler": "tpe", "multivariate": True, "constant_liar": True, "seed": seed},
    )


# ── trial attrs（tell 前写入——tell 后终态不可更，#172 实测）───────────────────


def set_trial_attrs_before_tell(study: Any, number: int, attrs: dict[str, Any]) -> None:
    """对在途 trial 写 user_attrs（须先于 tell——optuna ``check_trial_is_updatable``
    语义天然 fail-loud 已终态）。storage 面为 optuna 存储接口公开契约，经
    ``study._storage`` 取用（无公开 accessor，dashboard 等生态同款用法）。"""
    storage = study._storage
    trial_id = storage.get_trial_id_from_study_id_trial_number(study._study_id, number)
    for key, value in attrs.items():
        storage.set_trial_user_attr(trial_id, key, value)


def finalize_running_as_fail(study: Any, *, reason: str) -> int:
    """在途 RUNNING 一律终局化 FAIL（重建终局化 #175/#180 R4；Lier 停止避让、
    FAIL 不进拟合）。attrs（原因）先于 tell 写入。返回终局化条数。"""
    import optuna

    n = 0
    for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        set_trial_attrs_before_tell(study, trial.number, {ATTR_REASON: reason})
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        n += 1
    return n


# ── 开局三形态（init / continue / rebuild，#180 R1 + #182 修订）───────────────


def init_from_history(
    *,
    db_path: Path,
    df: pd.DataFrame,
    target_column: str,
    dims: dict[str, SpaceDim],
    direction: str,
    seed: int,
    model_run_id: str,
    band: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """从湿实验历史 parquet 建新 study：行 → COMPLETE trial（实测即数据）。

    20 门 = 健康地板（#183 R4）；schema 同构契约（搜索维 ⊆ 历史特征列）fail-loud。
    """
    import optuna

    if db_path.exists():
        raise FileExistsError(f"study db 已存在（撞名 fail-loud——继续既有 study 请直传该 db 路径）：{db_path}")
    assert_min_history(len(df))
    if target_column not in df.columns:
        raise ValueError(f"目标列 '{target_column}' 不在湿历史——列：{list(df.columns)}")
    feature_schema = [c for c in df.columns if c != target_column]
    unknown = sorted(set(dims) - set(feature_schema))
    if unknown:
        raise ValueError(f"搜索维不在湿历史 schema（schema 同构契约，#179）：{unknown}——历史特征列：{feature_schema}")

    # 行预检 + 值编码先行（红线 3：校验不过 db 零落盘——不留半建工件）
    dists = distributions_of(dims)
    prepared: list[tuple[dict[str, Any], float, dict[str, Any]]] = []
    for idx, row in df.iterrows():
        value = row[target_column]
        # 数值且非 NaN（NaN != NaN 自反不等）——pandas 标量谓词走纯 python 避开 stub 双义
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
            raise ValueError(f"湿历史第 {idx} 行目标列 '{target_column}' 非数值实测：{value!r}")
        params = {name: row[name] for name in dims}
        for name, v in params.items():
            if isinstance(v, float) and v != v:  # NaN 自反不等（纯 python 谓词）
                raise ValueError(f"湿历史第 {idx} 行搜索维 '{name}' 缺值（NaN）")
        assert_params_in_distributions(dists, params, where=f"湿历史第 {idx} 行")
        prepared.append((params, encode_objective(direction, float(value), band), {c: row[c] for c in df.columns}))

    study = create_study(db_path, direction=direction)
    set_identity_attrs(
        study,
        model_run_id=model_run_id,
        target_column=target_column,
        feature_schema=feature_schema,
        dims=dims,
        direction=direction,
        seed=seed,
    )

    non_search = [c for c in feature_schema if c not in dims]
    for params, encoded, row in prepared:
        raw = float(row[target_column])
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=encoded,
                params=params,
                distributions=dists,
                user_attrs={
                    ATTR_MEASURED: raw,
                    ATTR_CONDITIONS: {c: row[c] for c in non_search if not pd.isna(row[c])},
                },
            )
        )
    return {"action": "init", "study_db": str(db_path), "n_history": len(df), "feature_schema": feature_schema}


def rebind_model(study: Any, *, new_run_id: str) -> dict[str, Any]:
    """换模型开局（#182 R3）：归档旧信号态 + 更新 model.run_id + 清零 learning_signal
    一气呵成；重绑已归档 run_id（回滚旧模型）= 覆盖该键归档、重新计数。"""
    attrs = study.user_attrs
    old_run_id = attrs[ATTR_MODEL]["run_id"]
    archive = dict(attrs.get(ATTR_SIGNAL_ARCHIVE) or {})
    if ATTR_SIGNAL in attrs:
        archive[old_run_id] = attrs[ATTR_SIGNAL]
    # 重绑已归档 run_id（回滚旧模型，#182 R3）：覆盖该键归档、重新计数——旧纪元
    # 归档清出，本纪元关闭时以新计数覆写该键
    archive.pop(new_run_id, None)
    last_number = max((t.number for t in study.get_trials(deepcopy=False)), default=-1)
    fresh = {
        "epoch_start": last_number + 1,
        "positive": 0,
        "negative": 0,
        "positive_triggered": False,
        "negative_triggered": False,
        "updated_at_trial": last_number,
    }
    study.set_user_attr(ATTR_SIGNAL_ARCHIVE, archive)
    study.set_user_attr(ATTR_MODEL, {"run_id": new_run_id})
    study.set_user_attr(ATTR_SIGNAL, fresh)
    return {"archived_run_id": old_run_id, "reset": fresh}


def params_key(params: dict[str, Any]) -> str:
    """trial 参数 → 分组键（在途去重与推荐归组共用单源——排序键 + ensure_ascii 保持
    非类别值取值聚合稳定）。"""
    return json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _backfill_new_dims(
    trials: list[Any],
    old_dims: dict[str, SpaceDim],
    new_dims: dict[str, SpaceDim],
    history_df: pd.DataFrame | None,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """新增搜索维补值（#180 R4）：COMPLETE trial 的该维值 = conditions attr →
    湿历史按旧搜索维精确匹配（多行同值取值、多行异值/无匹配 fail-loud）→ 取不到
    fail-loud 列出冲突 trial。返回（trial number → 补值 dict, 新维名清单）。"""
    new_names = [n for n in new_dims if n not in old_dims]
    if not new_names:
        return {}, []
    old_names = list(old_dims)
    backfill: dict[int, dict[str, Any]] = {}
    conflicts: list[str] = []
    for trial in trials:
        values: dict[str, Any] = {}
        conditions = trial.user_attrs.get(ATTR_CONDITIONS) or {}
        for name in new_names:
            if name in conditions and not pd.isna(conditions[name]):
                values[name] = conditions[name]
                continue
            if history_df is not None:
                matched = history_df
                for old in old_names:
                    matched = matched[matched[old] == trial.params[old]]
                series = matched[name]
                assert isinstance(series, pd.Series)  # 单列选择恒 Series（stub 联合收窄）
                col = series.dropna().unique()
                if len(col) == 0:
                    conflicts.append(f"trial#{trial.number} 维 '{name}'：湿历史无旧参数匹配行")
                    continue
                if len(col) > 1:
                    conflicts.append(f"trial#{trial.number} 维 '{name}'：湿历史匹配多行异值 {list(map(str, col))}")
                    continue
                values[name] = col[0]
                continue
            conflicts.append(f"trial#{trial.number} 维 '{name}'：无 conditions attr 且未传湿历史补值")
        backfill[trial.number] = values
    if conflicts:
        raise ValueError(
            f"新增搜索维补值取不到（fail-loud，#180 R4）——冲突 {len(conflicts)} 条：\n" + "\n".join(conflicts)
        )
    return backfill, new_names


def rebuild_from_db(
    *,
    source_db: Path,
    target_db: Path,
    dims: dict[str, SpaceDim],
    direction: str,
    seed: int,
    model_run_id: str,
    history_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """迁移式开局（特征空间变更重建，#180 R4）：

    - COMPLETE 逐条 add_trial 保态迁移（state/value/attrs）；新增维度时 COMPLETE
      从 conditions attr / 湿历史实测列补值。
    - 新增维度时 PRUNED/FAIL 不迁（旧空间负面判断在新空间不可比，作废）；
      同空间重建（物理边界变）PRUNED/FAIL 照迁。
    - 越新边界 trial fail-loud 列名；在途 RUNNING 一律终局化 FAIL（原因 =
      rebuild_finalized，落源 db——旧 study 连同其信号归档留存盘上，#182 R2）。
    - 迁移报告随返回，不静默。
    """
    import optuna

    if target_db.resolve() == source_db.resolve():
        raise ValueError(f"重建目标与源 db 同路径（新 study_name 须不同）：{source_db}")
    if target_db.exists():
        raise FileExistsError(f"重建目标 db 已存在（撞名 fail-loud）：{target_db}")

    source = open_study(source_db)
    old_dims = parse_search_space(source.user_attrs[ATTR_SPACE])
    target_column = source.user_attrs[ATTR_TARGET]
    feature_schema = list(source.user_attrs[ATTR_SCHEMA])

    trials = source.get_trials(deepcopy=False)
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    others = [t for t in trials if t.state in (optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL)]
    assert_min_history(len(complete))  # db 锚口径：COMPLETE trial 数（#183 R4 两锚统一）

    backfill, new_names = _backfill_new_dims(complete, old_dims, dims, history_df)
    migrate_others = not new_names  # 新增维度时负面判断不迁（#180 R4）

    # 越界预检（fail-loud 列名，#180 R4）——先于源 db 任何变更（红线 3：校验不过
    # 源 db 零副作用，重试不带伤）
    dists = distributions_of(dims)
    for trial in complete:
        params = {**trial.params, **backfill.get(trial.number, {})}
        for name in params:
            if name not in dists:
                raise ValueError(f"trial#{trial.number} 参数 '{name}' 不在新空间维清单——候选：{sorted(dists)}")
        assert_params_in_distributions(dists, params, where=f"迁移 trial#{trial.number}")
    if migrate_others:
        for trial in others:
            assert_params_in_distributions(dists, trial.params, where=f"迁移 trial#{trial.number}")

    # 源 db 在途终局化（预检全过后——旧 study 留下一致终态）
    finalized = finalize_running_as_fail(source, reason="rebuild_finalized")

    new_study = create_study(target_db, direction=direction)
    set_identity_attrs(
        new_study,
        model_run_id=model_run_id,
        target_column=target_column,
        feature_schema=feature_schema + [n for n in new_names if n not in feature_schema],
        dims=dims,
        direction=direction,
        seed=seed,
    )

    n_complete = 0
    for trial in complete:
        params = {**trial.params, **backfill.get(trial.number, {})}
        new_study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=trial.value,
                params=params,
                distributions=dists,
                user_attrs=dict(trial.user_attrs),
            )
        )
        n_complete += 1
    n_others = 0
    if migrate_others:
        for trial in others:
            new_study.add_trial(
                optuna.trial.create_trial(
                    state=trial.state,
                    value=trial.value,
                    params=trial.params,
                    distributions=dists,
                    user_attrs=dict(trial.user_attrs),
                )
            )
            n_others += 1

    return {
        "action": "rebuild",
        "source_db": str(source_db),
        "study_db": str(target_db),
        "migration_report": {
            "migrated": {"complete": n_complete, "pruned_fail": n_others},
            "dropped": {
                "pruned_fail": len(others) - n_others,
                "reason": "新增维度——旧空间负面判断在新空间不可比（#180 R4）" if not migrate_others else None,
            },
            "finalized_running": finalized,
            "backfilled_dims": new_names,
        },
    }
