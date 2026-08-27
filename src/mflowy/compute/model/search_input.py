"""search_input handler —— Optuna 驱动的输入空间优化（inverse design）。

给定训练好的 model + 搜索空间约束，找出能让预测输出最优的输入特征组合。
核心抽象（``src.utils.study``）：``ParameterSearchSpace`` / ``get_sampler`` / ``search``。
模型加载与 ensemble 复用 ``loader`` + ``predict.ensemble_predict``。

设计契约：``model.y_names`` 是训练时定义的目标契约，search_input 优化全部 y_names，
方向由 ``directions`` dict 显式给出。不在推理时挑选「关心哪几个 y」——那是训练的事。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd

from mflowy.driver.handler import handler
from mflowy.middlewares.inject_df_or_none import inject_df_or_none
from mflowy.middlewares.log_search_input import log_search_input
from mflowy.utils.constants import RANDOM_STATE
from mflowy.utils.file import read_text
from mflowy.utils.path import split_path_to_py_with_target
from mflowy.utils.python_script_security_scan import scan_security
from mflowy.utils.study import (
    ContinuousSpace,
    DiscreteSpace,
    ParameterSearchSpace,
    get_sampler,
    search,
)

from .loader import loader as _loader
from .predict import ensemble_predict

logger = logging.getLogger(__name__)


def _infer_search_spaces(
    df: pd.DataFrame | None,
    x_names: list[str] | None = None,
) -> dict[str, ParameterSearchSpace]:
    """从真实 df 逐列推断搜索空间。``df=None`` 时返回 ``{}``（无上游 LOAD 步）。

    - int 列 → ``ContinuousSpace(int_min, int_max)``（``step=None``，``suggest_params`` 见 int 自动 ``step=1``）
    - float 列 → ``ContinuousSpace(float_min, float_max)``（连续）
    - bool 列 → ``DiscreteSpace([True, False])``
    - object/string 列 → ``DiscreteSpace(unique_values)``
    - 其他（datetime/category 等）→ ``logger.warning`` 跳过

    ``x_names`` 给定时只推断其中的列——避免把 target 等 model.x_names 之外的列
    误推断为搜索空间（trial 采样后 model.predict 会报 feature_names mismatch）。
    """
    if df is None or df.empty:
        return {}

    spaces: dict[str, ParameterSearchSpace] = {}
    for col in df.columns:
        if x_names is not None and col not in x_names:
            continue
        match df[col].dtype.kind:
            case "i" | "u":  # 整数（numpy + pandas Int*）
                desc = df[col].describe()
                spaces[col] = ContinuousSpace(int(desc["min"]), int(desc["max"]))
            case "f":  # 浮点（numpy + pandas Float*）
                desc = df[col].describe()
                spaces[col] = ContinuousSpace(float(desc["min"]), float(desc["max"]))
            case "b":  # 布尔（numpy bool + pandas Boolean）
                spaces[col] = DiscreteSpace([True, False])
            case "O" | "U" | "S":  # object / string
                spaces[col] = DiscreteSpace(list(df[col].dropna().unique()))
            case _:
                logger.warning(f"跳过列 '{col}'，dtype={df[col].dtype} 不可推断为搜索空间")
    return spaces


def _load_rules_validator(cross_rules_source: str) -> Callable[[pd.DataFrame], bool]:
    """加载用户 ``validate(df) -> bool`` 脚本作为 trial 内跨列约束校验器。

    复用 ``python_loader`` 的 ``scan_security`` + ``exec`` 模式。返回的 callable 接收
    单行 X_row DataFrame，返回 True 表示组合可行，False 触发 ``TrialPruned``。
    抛异常由调用方处理（不捕获，step-fail）。
    """
    _path, func = split_path_to_py_with_target(cross_rules_source)
    func = func or "validate"
    if not _path.exists():
        raise FileNotFoundError(f"cross_rules_source 文件不存在: {cross_rules_source}")

    code = read_text(_path)
    scan_security(code, func_name=func, returns=bool, args={"df": pd.DataFrame})

    exec_globals: dict[str, Any] = {"pd": pd, "__builtins__": __builtins__}
    exec(compile(code, "<cross_rules_source>", "exec"), exec_globals)

    validate_fn = exec_globals.get(func)
    if not callable(validate_fn):
        raise TypeError("validate 不是一个可调用对象")
    return validate_fn  # type: ignore


@handler(inject_df_or_none, log_search_input)
def search_input(
    df: pd.DataFrame | None,
    flavor: Annotated[Literal["XGB", "LGBM", "CAT", "RF", "MLP"], "模型模块名（训练时 @handler 注册的函数名）"],
    run_id: Annotated[str, "训练 model 步骤所在 Run 的 id"],
    directions: Annotated[
        dict[str, Literal["maximize", "minimize"]], "每个 y_name 的优化方向，key 必须完全等于 model.y_names"
    ],
    columns: Annotated[dict[str, ParameterSearchSpace] | None, "列约束（逐列覆盖推断的搜索空间）"] = None,
    cross_rules: Annotated[
        str | None, "/path/to/rulespy:validate, 跨列约束脚本路径，validate需要满足(df)->bool"
    ] = None,
    n_trials: Annotated[int, "Optuna trial 数量"] = 10000,
    random_seed: Annotated[int, "随机种子"] = RANDOM_STATE,
) -> pd.DataFrame:
    """model.search_input：搜索让所有 ``model.y_names`` 按 ``directions`` 最优的输入特征组合。"""
    import optuna

    # 1. 加载 model
    model_loader = _loader(flavor=flavor, run_id=run_id)
    fold_wrappers = list(model_loader)
    sample_wrapper = fold_wrappers[0]
    y_names = list(sample_wrapper.model.y_names)
    x_names = list(sample_wrapper.model.x_names)

    # 2. 构建搜索空间：infer + columns 列级合并（infer 限定在 x_names 内，排除 target 等）
    search_space: dict[str, ParameterSearchSpace] = {
        **_infer_search_spaces(df, x_names=x_names),
        **(columns or {}),
    }

    if not search_space:
        raise ValueError("搜索空间为空——df 无可推断列且未提供 columns")

    # 3. 早校验 directions 与 y_names 精确匹配
    directions_keys = set(directions.keys())
    y_names_set = set(y_names)
    if directions_keys != y_names_set:
        missing = y_names_set - directions_keys
        extras = directions_keys - y_names_set
        raise ValueError(
            f"directions 的 key 必须完全等于 model.y_names {y_names}；缺失={sorted(missing) or '无'}, 多余={sorted(extras) or '无'}"
        )
    directions_list = [directions[y] for y in y_names]  # 按 y_names 序展开

    # 4. 早校验 search_space 完全覆盖 x_names
    missing_x = set(x_names) - set(search_space.keys())
    if missing_x:
        raise ValueError(f"搜索空间未覆盖 model.x_names 的列: {sorted(missing_x)}")
    extras_x = set(search_space.keys()) - set(x_names)
    if extras_x:
        logger.warning(f"搜索空间包含 model.x_names 之外的列（model 将忽略）: {sorted(extras_x)}")

    # 5. 加载 cross_rules validator（可选）
    rules_fn: Callable[[pd.DataFrame], bool] | None = (
        _load_rules_validator(cross_rules) if cross_rules is not None else None
    )

    # 6. objective 闭包：search() 内部已调 suggest_params 拆 kwargs，返回 tuple 给多目标 study
    def objective(trial, **params):  # type: ignore[no-untyped-def]
        X_row = pd.DataFrame([params])
        if rules_fn is not None and not rules_fn(X_row):
            raise optuna.TrialPruned()
        preds = ensemble_predict(model_loader, X_row)
        return tuple(float(np.ravel(preds[y])[0]) for y in y_names)

    # 7. 委托 utils.study.search 跑 study（directions 长度 1 = 单目标，>1 = 多目标）
    study = search(
        param_space=search_space,
        objective=objective,
        n_trials=n_trials,
        sampler=get_sampler(seed=random_seed),
        directions=directions_list,
    )

    # 8. 输出 complete trials 的干净 input→output df
    rows: list[dict[str, Any]] = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = dict(t.params)
            for i, y in enumerate(y_names):
                row[y] = t.values[i]  # Optuna: 单/多目标统一用 trial.values
            rows.append(row)

    if not rows:
        logger.warning("所有 trial 都被 prune 或 n_trials=0，输出空 DataFrame")
        return pd.DataFrame()

    print(f"search_input: {len(rows)} complete trials, y={y_names}")
    if len(directions_list) == 1:
        best = study.best_trial
        print(f"best {y_names[0]}={best.values[0]:.4f}, params={best.params}")
    return pd.DataFrame(rows)
