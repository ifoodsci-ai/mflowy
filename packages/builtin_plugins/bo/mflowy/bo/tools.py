"""mflowy bo 域 MCP 工具 —— 六件全无状态（#180 R1/#184 定名）。

每次调用从 study sqlite db 加载状态；恒本地执行（#183：ask 恒本地、预测走既有
``model.predict``——bo 工具零触碰模型，耦合只发生在 agent 的 MCP 编排层）。
study db 路径 = 全工具锚（``<task_dir>/bo_studies/<study_name>/study.db``；伴随
工件——ask 候选 csv / 台账 round_00N.csv——落 db 同目录）。

一轮两相主循环（agent 编排，#179 R1 六步）：``bo_ask`` 出完整 schema 候选 csv
（无预测列、非搜索维空列）→ agent 填钉值 → 既有 ``predict`` 补点值 → agent 读卡
逐行判 keep/prune → ``bo_prune`` / ``bo_diversity_check`` → agent 推台账
``round_00N.csv`` → 湿实验（跨天人肉）→ ``bo_tell`` 批量回填（信号批末附带）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from pydantic import Field

from . import study as _study
from .diversity import farthest_point_selection
from .signal import compute_learning_signal, signal_suggestion
from .study import (
    ATTRIBUTION_MAP,
    ATTR_CONDITIONS,
    ATTR_DIRECTION,
    ATTR_LABEL,
    ATTR_MEASURED,
    ATTR_REASON,
    ATTR_REPLICATE_OF,
    ATTR_ROUND,
    ATTR_SCHEMA,
    ATTR_SIGNAL,
    ATTR_SOURCE,
    ATTR_SPACE,
    LABELS,
    PRUNED_REASONS,
    SOURCES,
)
from .types import SpaceDim


def _open_with_dims(study_db: str) -> tuple[Any, dict[str, SpaceDim], Path]:
    """开 study + 空间组（attrs 单源）+ db 路径。"""
    path = Path(study_db)
    st = _study.open_study(path, dims=None)
    dims = _study.parse_search_space(st.user_attrs[_study.ATTR_SPACE])
    return st, dims, path


def _running_trial(st: Any, number: int) -> Any:
    """按号取在途 trial；不存在/已终态 fail-loud 带现终态与 attrs（#180 R5）。"""
    for t in st.get_trials(deepcopy=False):
        if t.number == number:
            if not str(t.state).endswith("RUNNING"):
                raise ValueError(
                    f"trial#{number} 已终态 {t.state}（attrs={dict(t.user_attrs)}）"
                    "——两路回填先到为准、无覆盖入口（#181 R3）"
                )
            return t
    raise ValueError(f"trial#{number} 不存在于 study——台账序号 = trial number 直用，请对账台账 trial 列")


def _check_conditions(st: Any, conditions: dict[str, Any]) -> None:
    schema = set(st.user_attrs[ATTR_SCHEMA])
    unknown = sorted(set(conditions) - schema)
    if unknown:
        raise ValueError(f"conditions 含 schema 外列（schema 同构契约，#179）：{unknown}——特征列：{sorted(schema)}")


def _check_label(label: Any) -> None:
    if label not in LABELS:
        raise ValueError(f"label 须为 {'/'.join(LABELS)}——收到 {label!r}（与 db attrs 同源枚举，#181 R1）")


def _check_source(source: Any) -> None:
    if source not in SOURCES:
        raise ValueError(f"source 须为 {'/'.join(SOURCES)}——收到 {source!r}")


def _parse_band(band: list[float] | None) -> tuple[float, float] | None:
    if band is None:
        return None
    if len(band) != 2 or band[0] >= band[1]:
        raise ValueError(f"band 须为 [lo, hi] 且 lo < hi——收到 {band!r}")
    return float(band[0]), float(band[1])


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """tmp+rename 原子落盘（红线 3）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ── ① 开局：init / continue / rebuild（#180 R1 + #182 R5 修订）────────────────


def bo_load_or_init(
    study_name: Annotated[str, Field(description="study 名（决定落位 <task_dir>/bo_studies/<study_name>/study.db）")],
    model_run_id: Annotated[str, Field(description="预测模型 run_id（模型 = 调用入参非身份——换模型仅重置学习信号，#182）")],
    data: Annotated[
        str | None,
        Field(description="湿实验历史 parquet 绝对路径（开局锚一；无 db 时 = 从历史建新 study）"),
    ] = None,
    db: Annotated[
        str | None,
        Field(description="既有 study db 绝对路径（开局锚二：续用或迁移式重建的源；其他五工具的 study_db 即此路径"),
    ] = None,
    target_column: Annotated[
        str | None, Field(description="目标列名（direction 键目标列；data 锚开局必填）")
    ] = None,
    direction: Annotated[
        str | None,
        Field(description="业务方向 maximize|minimize|match（空间组同一性字段；data 锚必填；与 search_space 一并传以比对指纹）"),
    ] = None,
    search_space: Annotated[
        dict[str, dict] | None,
        Field(
            description="搜索空间组（BOCard 空间组逐维直传——agent 读卡：{维名: {kind: continuous|discrete|combinatorial, ...payload}}；"
            "传入即比对指纹，不符触发迁移式重建；缺省 = 续用不比对"
        ),
    ] = None,
    band: Annotated[
        list[float] | None, Field(description="目标带 [lo, hi]（direction=match 时必填——hinge 编码；策略组可后改不重建）")
    ] = None,
    seed: Annotated[int | None, Field(description="TPE 种子（同一性字段；缺省开局 42、续用沿用 db 内种子）")] = None,
) -> dict:
    """开局 / 重进 bo study（三形态一入口：init ← 历史 parquet；continue ← 既有 db；
    rebuild ← db 迁移式重建）。

    何时使用 bo_load_or_init：
    - 首次开局：data（湿实验历史 ≥20 行，#183 机器门）+ target_column + direction +
      search_space → 建新 study（历史行 = COMPLETE trial）
    - 重进/换模型：db（可带新 run_id → 学习信号归档重置一气呵成）
    - 特征空间变更（空间组/direction/seed 变）：db + search_space + direction（+
      data 补新增维值）→ 迁移式重建（COMPLETE 保态迁移、在途终局化 FAIL、迁移报告
      不静默）

    data 与 db 至少一个（双面拦截机器面，#182）；卡空间维 ⊆ model.x_names 核对与
    分类模型拒绝 = 剧本问句无机器门（#179 R1）。
    """
    if data is None and db is None:
        raise ValueError(
            "开局需 data（湿实验历史 parquet）或 db（既有 study）至少一个"
            "（#182 双面拦截机器面）——两缺即拒；剧本面预检同面"
        )
    if direction is not None and direction not in ("maximize", "minimize", "match"):
        raise ValueError(f"direction 须为 maximize|minimize|match——收到 {direction!r}")

    task_root = _task_root_of(data, db)
    target_db = _study.study_root(task_root, study_name) / "study.db"

    if db is None:
        # ── init：历史 parquet → 新 study ─────────────────────────────────────
        missing = [k for k, v in {"target_column": target_column, "direction": direction, "search_space": search_space}.items() if v is None]
        if missing:
            raise ValueError(f"data 锚开局必填 {missing}（target_column/direction/search_space）")
        if direction != "match" and band is not None:
            raise ValueError(f"band 仅 direction=match 需要——收到 {band!r}（构造性隔离：软约束唯一形态 = band 标量）")
        dims = _study.parse_search_space(search_space)
        history = _read_history(data)
        report = _study.init_from_history(
            db_path=target_db,
            df=history,
            target_column=target_column,
            dims=dims,
            direction=direction,
            seed=seed if seed is not None else 42,
            model_run_id=model_run_id,
            band=_parse_band(band) if direction == "match" else None,
        )
        report["signal"] = None
        return report

    # ── db 锚：continue / rebuild ────────────────────────────────────────────
    source_path = Path(db)
    st = _study.open_study(source_path)
    _require_identity_attrs(st)
    if search_space is None:
        if data is not None:
            raise ValueError(
                "db 锚续用不收 data（意图不明）——续用只传 db；特征空间变更重建请一并传 search_space + direction（data 可作新增维补值源）"
            )
        _study.assert_min_history(
            sum(1 for t in st.get_trials(deepcopy=False) if str(t.state).endswith("COMPLETE"))
        )
        out: dict[str, Any] = {"action": "continue", "study_db": str(source_path)}
        if st.user_attrs[_study.ATTR_MODEL]["run_id"] != model_run_id:
            out["run_id_rebind"] = _study.rebind_model(st, new_run_id=model_run_id)
        out["signal"] = st.user_attrs.get(ATTR_SIGNAL)
        return out

    # 指纹比对（同一性 = 空间组 + direction + seed，#178 M1）
    if direction is None:
        raise ValueError("传入 search_space 即比对指纹——direction 须一并传入（空间组同一性字段）")
    dims = _study.parse_search_space(search_space)
    effective_seed = seed if seed is not None else st.user_attrs[_study.ATTR_ENGINE]["seed"]
    fingerprint = _study.fingerprint_of(dims, direction=direction, seed=effective_seed)
    if fingerprint == st.user_attrs[_study.ATTR_FINGERPRINT]:
        if data is not None:
            raise ValueError("指纹相符（续用）不收 data——续用只传 db；重建场景指纹必不符，请核对空间组/direction/seed")
        _study.assert_min_history(
            sum(1 for t in st.get_trials(deepcopy=False) if str(t.state).endswith("COMPLETE"))
        )
        out = {"action": "continue", "study_db": str(source_path)}
        if st.user_attrs[_study.ATTR_MODEL]["run_id"] != model_run_id:
            out["run_id_rebind"] = _study.rebind_model(st, new_run_id=model_run_id)
        out["signal"] = st.user_attrs.get(ATTR_SIGNAL)
        return out

    # ── rebuild：迁移式重建（特征空间变更，#180 R4）─────────────────────────
    return _study.rebuild_from_db(
        source_db=source_path,
        target_db=target_db,
        dims=dims,
        direction=direction,
        seed=effective_seed,
        model_run_id=model_run_id,
        history_df=_read_history(data) if data is not None else None,
    )


def _task_root_of(data: str | None, db: str | None) -> Path:
    """开局锚依次回退（#180 R2）：data → db；``bo_studies/<name>/study.db`` 布局
    解包两级还原 task_dir。"""
    for anchor in (data, db):
        if anchor is not None:
            d = Path(anchor).absolute().parent
            return d.parent if d.name == "bo_studies" else d
    raise ValueError("data 与 db 均缺（不可达——入口已拦）")  # pragma: no cover


def _read_history(data: str) -> pd.DataFrame:
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"湿实验历史 parquet 不存在：{path}")
    return pd.read_parquet(path)


def _require_identity_attrs(st: Any) -> None:
    missing = [
        key
        for key in (
            _study.ATTR_MODEL,
            _study.ATTR_TARGET,
            _study.ATTR_SCHEMA,
            _study.ATTR_SPACE,
            _study.ATTR_DIRECTION,
            _study.ATTR_FINGERPRINT,
            _study.ATTR_ENGINE,
        )
        if key not in st.user_attrs
    ]
    if missing:
        raise ValueError(f"study db 缺身份键 {missing}——非 bo_load_or_init 产物（db = 全部状态的唯一真源）")


# ── ② 推荐：ask（完整 schema 候选 csv，无预测列——#179 R1）──────────────────


def bo_ask(
    study_db: Annotated[str, Field(description="study db 绝对路径（bo_load_or_init 返回的 study_db）")],
    n_candidates: Annotated[
        int,
        Field(
            description="候选数（批；k×批缺省直传本参数——k agent 自判，卡零新字段，#183 R6）",
            ge=1,
        ),
    ],
) -> dict:
    """推荐一批候选（study.ask）→ 完整 schema 候选 csv：搜索维 = 采样值、非搜索维
    空列（空列在场优于列缺省——agent 面从「记着补哪些列」降为「填空」，#179 R1）。

    何时使用 bo_ask：
    - 推荐幕第一步：出候选 → agent 填钉值 → 既有 model.predict 补点值 → 读卡剪枝
    - 候选 csv 落 db 同目录 ``ask_<first>_<last>.csv``（过程工件——agent 据此推导
      推送台账 round_00N.csv，台账另含回填/回写列）

    在途避让：对 RUNNING trial（跨轮在途 + 批内）精确去重重采样（撞点落 FAIL/
    diversity_rejected）；纯 Virtual 局 = 回填缺席的用法，多轮 ask 靠在途避让铺开
    （#183 R3）。虚拟预测值永不入 study。
    """
    import optuna

    st, dims, path = _open_with_dims(study_db)
    dists = _study.distributions_of(dims)
    running: set[str] = {
        _study._params_key(t.params)
        for t in st.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
    }
    schema: list[str] = list(st.user_attrs[ATTR_SCHEMA])

    picked: list[tuple[int, dict[str, Any]]] = []
    dropped = 0
    attempts, max_attempts = 0, 4 * n_candidates
    while len(picked) < n_candidates and attempts < max_attempts:
        attempts += 1
        trial = st.ask(fixed_distributions=dists)
        key = _study._params_key(trial.params)
        if key in running:
            _study.set_trial_attrs_before_tell(st, trial.number, {ATTR_REASON: "diversity_rejected"})
            st.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            dropped += 1
            continue
        running.add(key)
        picked.append((trial.number, dict(trial.params)))

    csv_path: str | None = None
    if picked:
        df = pd.DataFrame(
            [{**{c: None for c in schema}, **dict(params), "trial": number} for number, params in picked]
        )[["trial", *[c for c in schema if c not in ("trial",)]]]
        csv_path = str(path.parent / f"ask_{picked[0][0]}_{picked[-1][0]}.csv")
        _write_csv_atomic(df, Path(csv_path))
    return {
        "study_db": str(path),
        "n_asked": len(picked),
        "duplicates_dropped": dropped,
        "candidates_csv": csv_path,
        "trials": [{"trial": n, "params": p} for n, p in picked],
    }


# ── ③ 剪枝：tell(PRUNED) + 原因（提案期；求值归 agent——#180 R1）──────────────


def bo_prune(
    study_db: Annotated[str, Field(description="study db 绝对路径")],
    trials: Annotated[list[int], Field(description="拟剪枝的候选 trial number 清单（bo_ask 响应的 trial）", min_length=1)],
    reason: Annotated[
        str,
        Field(description=f"负面判断原因（与台账/attrs 同源词表）：{'/'.join(PRUNED_REASONS)}"),
    ],
) -> dict:
    """剪枝（提案期）：候选落 PRUNED + 原因 attr——负面证据进 TPE 坏组密度
    （#177 R2 实测：弃用落 PRUNED 合理）。

    何时使用 bo_prune：
    - agent 读卡 + 预测值逐行判 prune 后（硬规则违例 / 预测出带 / 用户拒绝），
      将被弃候选一次性终局化；范围×目标带 = 相交判定（无交集才出带，骑带缘归
      agent 裁量——#178 M2），求值全部在 agent 侧
    """
    import optuna

    if reason not in PRUNED_REASONS:
        raise ValueError(f"reason 须为 {'/'.join(PRUNED_REASONS)}——收到 {reason!r}")
    st, _, path = _open_with_dims(study_db)
    for number in trials:
        _running_trial(st, number)
        _study.set_trial_attrs_before_tell(st, number, {ATTR_REASON: reason})
        st.tell(number, state=optuna.trial.TrialState.PRUNED)
    return {"study_db": str(path), "pruned": list(trials), "reason": reason}


# ── ④ 多样性落选：FPS 选批 + 落选 tell(FAIL)（#178 R3/#184）──────────────────


def bo_diversity_check(
    study_db: Annotated[str, Field(description="study db 绝对路径")],
    candidates: Annotated[list[int], Field(description="agent 判 keep 后的候选 trial number 清单（升序传入）", min_length=1)],
    batch_size: Annotated[int, Field(description="本轮批大小（卡批缺省由 agent 读数传入；实验室容量，可逐轮覆写）", ge=1)],
) -> dict:
    """多样性筛选：归一化 farthest-point-selection 从 kept 候选选多样批；落选者
    落 FAIL + diversity_rejected（冗余 ≠ 负面判断——落 PRUNED 会把 TPE 推离好区域，
    且落选条件下轮可再推，#178 批注）。

    何时使用 bo_diversity_check：
    - 剪枝后批内仍多于实验室容量时收口多样批（归一化按空间 spec：continuous
      min-max、log 先取对数、discrete 恒等/汉明——度量 v1 固定默认 = 实现层常量）
    """
    import optuna

    st, dims, path = _open_with_dims(study_db)
    frozen = [(number, _running_trial(st, number).params) for number in candidates]
    selected_idx, rejected_idx = farthest_point_selection(
        dims, [params for _, params in frozen], batch_size=batch_size
    )
    selected = [frozen[i][0] for i in selected_idx]
    rejected = [frozen[i][0] for i in rejected_idx]
    for number in rejected:
        _study.set_trial_attrs_before_tell(st, number, {ATTR_REASON: "diversity_rejected"})
        st.tell(number, state=optuna.trial.TrialState.FAIL)
    return {
        "study_db": str(path),
        "selected": selected,
        "rejected": rejected,
        "batch_size": batch_size,
    }


# ── ⑤ 回填：唯一湿实验期批量写入口（信号批末附带——#180 R1/#177 R4）──────────

_TELL_DOC_ROWS = """行形态（list[dict]，单行为特例；寻址 trial 与 params 二选一、都给报错）：
- done：{"trial": n, "measured_value": v, "label": close|severe_mismatch|not_counted?,
  "round"?: int, "conditions"?: {非搜索特征列: 值}} —— COMPLETE + 值 + 标签 + 来源
  （推荐 trial 的来源恒 recommended）；label 可缺（缺 = 未判定，不进信号）
- failed：{"trial": n, "attribution": conditional|operational|dirty_data} ——
  conditional → PRUNED + 标签 severe_mismatch（计负向）；operational/dirty_data
  → FAIL（不计信号）
- deviated：{"trial": n, "params": {实测参数}, "measured_value": v, "variant": bool,
  "label"?, "conditions"?} —— 新 trial 追加（实测参数入档）；variant=true 关原
  trial（FAIL/variant_replaced）、新行来源 recommended，否则原 trial 留在途、新行
  来源 spontaneous（#177 R3 变体确认语义）
- abandoned：{"trial": n} —— FAIL + abandoned（物流事实，作废无判断）
- appended（追加/平行样余条）：{"params": {...}, "measured_value": v,
  "source"?: recommended|spontaneous（缺省 spontaneous）, "replicate_of"?: 原推荐
  trial 号, "label"?, "round"?, "conditions"?} —— add_trial COMPLETE；平行样 =
  首条 done + 余条 appended(replicate_of, source=recommended)"""


def bo_tell(
    study_db: Annotated[str, Field(description="study db 绝对路径")],
    rows: Annotated[
        list[dict],
        Field(description="批量回填行（一批台账多行 = 一次调用；行形态见 docstring），一批一次 bo_tell、学习信号批末报告不打断"),
    ],
    band: Annotated[
        list[float] | None,
        Field(description="目标带 [lo, hi]（direction=match 的 study 必填——实测值 hinge 编码；agent 从卡读数，策略组可后改不重建）"),
    ] = None,
    positive_gate: Annotated[int, Field(description="正向累计门（卡缺省 100；agent 从卡读数传入）", ge=1)] = 100,
    negative_gate: Annotated[int, Field(description="负向累计门（卡缺省 3）", ge=1)] = 3,
    window_k: Annotated[int | None, Field(description="小窗口门 K（最近 K 条推荐；缺省关——启用须与 window_m 同传且 m < 负向门）", ge=1)] = None,
    window_m: Annotated[int | None, Field(description="小窗口门 m（最近 K 条里 m 条负向即触发）", ge=1)] = None,
) -> dict:
    """回填（唯一湿实验期写入口，批量）：写 user_attrs + tell 单一封装——attrs
    先于 tell（#172 实测：tell 后不可更）。判定标签 = agent 复算 predict 后判定
    （判定归 agent、负向归因用户确认后才 tell——#177 R3）；标签写入即事实。

    何时使用 bo_tell：
    - 湿实验结果回来后（台账更新 / 带序号消息两路，agent 对账后批量调用）；
      平行样逐条不合并（独立测量）；已终态 trial 再操作 fail-loud（先到为准）
    - 批末一次学习信号检查（推荐级计数制）：未触发静默；触发返回门禁与建议
      （继续推荐 / 交棒更新模型——人点火），不拦截批继续；agent 无主动查询面

    """ + _TELL_DOC_ROWS
    import optuna

    if (window_k is None) != (window_m is None):
        raise ValueError("窗口门 window_k/window_m 须成对传入（缺省关）")
    if window_m is not None and window_m >= negative_gate:
        raise ValueError(f"window_m 须小于 negative_gate（否则累计门先响、窗口门冗余）——收到 {window_m} ≥ {negative_gate}")

    st, dims, path = _open_with_dims(study_db)
    dists = _study.distributions_of(dims)
    direction = st.user_attrs[ATTR_DIRECTION]
    parsed_band = _parse_band(band)
    if direction == "match" and parsed_band is None:
        raise ValueError("direction=match 的 study 回填需 band（实测值 hinge 编码）——agent 从卡读数传入")

    told: list[dict[str, Any]] = []
    for row in rows:
        told.append(_apply_tell_row(st, dists, direction, parsed_band, row))

    state = compute_learning_signal(
        st,
        positive_gate=positive_gate,
        negative_gate=negative_gate,
        window_k=window_k,
        window_m=window_m,
    )
    st.set_user_attr(ATTR_SIGNAL, state)
    out: dict[str, Any] = {"study_db": str(path), "told": told}
    suggestion = signal_suggestion(state)
    if suggestion is not None:  # 未触发静默（#180 R1）
        out["signal"] = {"state": state, "suggestion": suggestion}
    return out


def _apply_tell_row(
    st: Any,
    dists: dict[str, Any],
    direction: str,
    band: tuple[float, float] | None,
    row: dict[str, Any],
) -> dict[str, Any]:
    """单行回填（结构 fail-loud——未知 status/缺字段/终态冲突在此拦）。"""
    import optuna

    status = row.get("status")
    number = row.get("trial")
    params = row.get("params")
    # 寻址二选一（#180 R1）；deviated 例外——trial = 原推荐（处置对象）、params = 实测参数（新 trial）
    if number is not None and params is not None and status != "deviated":
        raise ValueError(f"寻址二选一（#180 R1）——trial 与 params 同时给出：{row}")
    if number is None and params is None:
        raise ValueError(f"回填行缺寻址（trial 或 params）：{row}")

    def _common_attrs() -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        if row.get("label") is not None:
            _check_label(row["label"])
            attrs[ATTR_LABEL] = row["label"]
        if row.get("round") is not None:
            attrs[ATTR_ROUND] = int(row["round"])
        if row.get("conditions") is not None:
            _check_conditions(st, row["conditions"])
            attrs[ATTR_CONDITIONS] = row["conditions"]
        return attrs

    def _require_measured() -> float:
        value = row.get("measured_value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"measured_value 须为数值——收到 {value!r}：{row}")
        return float(value)

    # ── trial 寻址四形态 ─────────────────────────────────────────────────────
    if number is not None:
        _running_trial(st, number)
        if status == "done":
            raw = _require_measured()
            attrs = {ATTR_MEASURED: raw, ATTR_SOURCE: "recommended", **_common_attrs()}
            _study.set_trial_attrs_before_tell(st, number, attrs)
            st.tell(number, values=_study.encode_objective(direction, raw, band))
            return {"trial": number, "status": "done", "state": "COMPLETE"}
        if status == "failed":
            attribution = row.get("attribution")
            if attribution not in ATTRIBUTION_MAP:
                raise ValueError(f"failed 行 attribution 必填 ∈ {sorted(ATTRIBUTION_MAP)}——收到 {attribution!r}")
            state_name, reason = ATTRIBUTION_MAP[attribution]
            attrs = {ATTR_REASON: reason, **_common_attrs()}
            if attribution == "conditional":  # 条件致因失败 = 严重不符（计负向，#180 R1）
                attrs[ATTR_LABEL] = "severe_mismatch"
                attrs[ATTR_SOURCE] = "recommended"
            _study.set_trial_attrs_before_tell(st, number, attrs)
            st.tell(
                number,
                state=getattr(optuna.trial.TrialState, state_name),
            )
            return {"trial": number, "status": "failed", "state": state_name}
        if status == "abandoned":
            _study.set_trial_attrs_before_tell(st, number, {ATTR_REASON: "abandoned"})
            st.tell(number, state=optuna.trial.TrialState.FAIL)
            return {"trial": number, "status": "abandoned", "state": "FAIL"}
        if status == "deviated":
            if not isinstance(params, dict) or not params:
                raise ValueError(f"deviated 行须带实测 params：{row}")
            raw = _require_measured()
            variant = bool(row.get("variant"))
            _study.assert_params_in_distributions(dists, params, where=f"deviated 新 trial（原 trial#{number}）")
            common = _common_attrs()
            st.add_trial(
                optuna.trial.create_trial(
                    state=optuna.trial.TrialState.COMPLETE,
                    value=_study.encode_objective(direction, raw, band),
                    params=dict(params),
                    distributions=dists,
                    user_attrs={
                        ATTR_MEASURED: raw,
                        ATTR_SOURCE: "recommended" if variant else "spontaneous",
                        **common,
                    },
                )
            )
            if variant:  # 推荐变体：关原推荐 trial（FAIL，#177 R3）
                _study.set_trial_attrs_before_tell(st, number, {ATTR_REASON: "variant_replaced"})
                st.tell(number, state=optuna.trial.TrialState.FAIL)
                original = "FAIL(variant_replaced)"
            else:
                original = "RUNNING（默认自发追加——原推荐留在途，#177 R3）"
            return {
                "trial": number,
                "status": "deviated",
                "new_trial": "add_trial",
                "original": original,
                "source": "recommended" if variant else "spontaneous",
            }
        if status is not None and status != "appended":
            raise ValueError(
                f"未知 status {status!r}——候选：done|failed|deviated|abandoned|appended（#181 R1 台账状态枚举同源）"
            )
        raise ValueError(f"trial 寻址行 status 须 ∈ done|failed|deviated|abandoned——收到 {status!r}：{row}")

    # ── params 寻址：追加 / 平行样余条（status 可缺省 appended）──────────────
    if status is not None and status != "appended":
        raise ValueError(f"params 寻址行 = 追加形态（status 可缺省/appended）——收到 {status!r}：{row}")
    raw = _require_measured()
    source = row.get("source", "spontaneous")
    _check_source(source)
    _study.assert_params_in_distributions(dists, params, where="追加行（超特征空间拒收并指明列，#177 R3）")
    attrs = {ATTR_MEASURED: raw, ATTR_SOURCE: source, **_common_attrs()}
    if row.get("replicate_of") is not None:
        attrs[ATTR_REPLICATE_OF] = int(row["replicate_of"])
    st.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=_study.encode_objective(direction, raw, band),
            params=dict(params),
            distributions=dists,
            user_attrs=attrs,
        )
    )
    return {"status": "appended", "state": "COMPLETE", "source": source}


# ── ⑥ 数据出口：db → 单 parquet（仅 COMPLETE，#182 R1）────────────────────────


def bo_export_dataset(
    study_db: Annotated[str, Field(description="study db 绝对路径")],
    output_path: Annotated[str, Field(description="导出 parquet 绝对路径（显式参数无缺省——按需主动调用，信号触发不自动导出）")],
) -> dict:
    """导出 study 数据集：db trials → 单个 parquet（``load.parquet`` 直吃，上游零新
    接入通道）。仅 COMPLETE（实测即数据，含结果很差的）；failed/PRUNED 不进数据集
    ——负信息留在台账与 db 供审计。

    何时使用 bo_export_dataset：
    - 学习信号触发后人点火重跑上游链（clean → modeling）前，把累计湿数据导出为
      训练集；列 = 条件 schema（搜索维 + conditions 钉值/实测条件）+ measured_value
      + round/replicate_of/source
    """
    import optuna

    st, _, path = _open_with_dims(study_db)
    schema = list(st.user_attrs[ATTR_SCHEMA])
    rows: list[dict[str, Any]] = []
    for trial in st.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        attrs = trial.user_attrs
        conditions = attrs.get(ATTR_CONDITIONS) or {}
        row: dict[str, Any] = {c: None for c in schema}
        row.update({k: v for k, v in conditions.items() if k in row})
        row.update(trial.params)
        row["measured_value"] = attrs.get(ATTR_MEASURED, trial.value)
        if attrs.get(ATTR_ROUND) is not None:
            row["round"] = attrs[ATTR_ROUND]
        if attrs.get(ATTR_REPLICATE_OF) is not None:
            row["replicate_of"] = attrs[ATTR_REPLICATE_OF]
        if attrs.get(ATTR_SOURCE) is not None:
            row["source"] = attrs[ATTR_SOURCE]
        rows.append(row)
    df = pd.DataFrame(rows, columns=[*schema, "measured_value", "round", "replicate_of", "source"])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return {"output_path": str(out), "n_rows": len(df), "columns": list(df.columns)}
