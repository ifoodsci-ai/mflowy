"""Optuna 学习/搜索共享抽象。"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic_core import core_schema

if TYPE_CHECKING:
    from optuna import Trial
    from optuna.pruners import BasePruner
    from optuna.samplers import BaseSampler
    from optuna.study import StudyDirection


@dataclass
class ContinuousSpace[T: int | float]:
    start: T
    end: T
    step: T | Literal["log"] | None = None


class DiscreteSpace[T: int | float | str | bool](list[T]):
    def __init__(self, iterable=()):
        super().__init__(iterable)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """yaml list 值 → DiscreteSpace 实例。

        pydantic 对 list 子类只做 is-instance 不构造（LIST_TYPES 精确匹配不含子类），
        故在 list schema 校验之后包一层构造——after 而非 before：before 构造的实例
        会被内层 list schema 重新摊平成 plain list。
        """
        args = get_args(source_type)
        return core_schema.no_info_after_validator_function(cls, handler(list[args[0]] if args else list))


type ParameterSearchSpace = ContinuousSpace | DiscreteSpace

# ── study 落库锚（#89 ①修正：JobProvider.hyperparameters_search 的 study_storage_path 通道） ──
# 非空时 search() 建 study 走该 sqlite + load_if_exists——同一路径重复搜索即增量
# 续跑（复用前史）；一库一 study（名 = 路径词干）。会话域内直连路径
# （_open_study，study_name = 结构键）不经此锚——两路径命名约定各自自洽。
_study_storage: ContextVar[str | None] = ContextVar("mflowy_study_storage", default=None)


def set_study_storage(path: str | None) -> Token[str | None]:
    """锚定 search() 的 study 落库位置（job_provider 线程内 set，用毕 reset——
    token 语义同 set_task_dir）。"""
    return _study_storage.set(path)


def reset_study_storage(token: Token[str | None]) -> None:
    _study_storage.reset(token)


def suggest_params(trial: Trial, param_space: dict[str, ParameterSearchSpace]) -> dict[str, Any]:
    """从 Optuna Trial 生成参数建议"""
    params = {}
    for name, space in param_space.items():
        if isinstance(space, DiscreteSpace):
            params[name] = trial.suggest_categorical(name, list(space))
        elif isinstance(space, ContinuousSpace):
            lo, hi = space.start, space.end
            if space.step == "log":
                log, step = True, None
            else:
                log, step = False, space.step

            if isinstance(lo, int):
                params[name] = trial.suggest_int(name, lo, hi, step=step or 1, log=log)
            else:
                params[name] = trial.suggest_float(name, lo, hi, step=step, log=log)
        else:
            raise ValueError(f"Invalid parameter space for {name}: {space}")
    return params


def min_resource_of(cap: int) -> int:
    """Hyperband min_resource 单源公式：保守取 cap//4（dl 封装与曲线检查点共用，防口径漂移）。"""
    return max(1, cap // 4)


def search[T: float | tuple[float, ...]](
    param_space: dict[str, ParameterSearchSpace],
    objective: Callable[..., T],
    n_trials: int,
    *,
    sampler: BaseSampler | None = None,
    timeout: float | None = None,
    pruner: BasePruner | None = None,
    direction: str | StudyDirection | None = None,
    directions: list[str | StudyDirection] | None = None,
):
    """纯骨架：sampler/pruner/directions 全透传，零策略零缺省——场景知识住在
    ``search_ml`` / ``search_dl`` / ``search_inverse`` 三封装，勿裸调本函数。

    Example:
        >>> space = {"lr": ContinuousSpace(1e-5, 1e-1, step="log")}
        >>> def train(trial, lr):
        ...     return lr * 100
        >>> study = search_ml(space, train, n_trials=50, direction="maximize")
    """

    def _objective(trial: Trial) -> T:
        params = suggest_params(trial, param_space)
        return objective(trial, **params)

    import optuna

    if path := _study_storage.get():
        # 落库锚生效（#89 ①修正）：sqlite + load_if_exists——同路径重复搜索复用
        # 前史（对同一模型多次 HPO 的增量续跑）；一库一 study，名 = 路径词干。
        study = optuna.create_study(
            study_name=Path(path).stem,
            storage=f"sqlite:///{path}",
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
            directions=directions,
        )
    else:
        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction=direction,
            directions=directions,
        )
    study.optimize(_objective, n_trials, timeout)
    return study


# ── 场景封装：搜索策略知识的居所（禁止裸调 search） ──────────────────────────


def search_ml(param_space, objective, n_trials, *, direction=None, timeout=None, seed: int | None = None):
    """传统 ML 超参搜索：tpe、无 pruner——K 折 trial 无逐折独立中间值，剪枝不诚实
    （显式 NopPruner：optuna 缺省 MedianPruner 会静默挂载，与"不假装"相悖）。
    seed 注入 sampler（会话增量 study 的目标同一性分量之一）。"""
    import optuna
    from optuna import samplers

    return search(
        param_space,
        objective,
        n_trials,
        sampler=samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner(),
        direction=direction,
        timeout=timeout,
    )


def search_dl(
    param_space, objective, n_trials, *, max_resource: int, direction=None, timeout=None, seed: int | None = None
):
    """资源轴增量评估的长训练（boosting staged / 未来 DL epoch 上报）：tpe + Hyperband 加速。

    min_resource 单源取 ``min_resource_of(max_resource)``；调用方须保证 objective
    沿资源轴上报中间值（无诚实中间值的路径请走 search_ml，不假装）。seed 注入
    sampler（测试确定性——剪枝断言对采样随机性敏感）。
    """
    import optuna
    from optuna import samplers

    lo = min_resource_of(max_resource)
    return search(
        param_space,
        objective,
        n_trials,
        sampler=samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.HyperbandPruner(min_resource=lo, max_resource=max_resource),
        direction=direction,
        timeout=timeout,
    )


def search_inverse(
    param_space, objective, n_trials, *, direction=None, directions=None, seed: int | None = None, timeout=None
):
    """模型逆向搜索（inverse optimization）：单目标 tpe / 多目标 NSGAII
    （pruner 与多目标不兼容——本场景恒无 pruner，显式 NopPruner）；seed 注入 sampler 保可复现。"""
    import optuna
    from optuna import samplers

    if directions and len(directions) > 1:
        sampler = samplers.NSGAIISampler(seed=seed)
    else:
        sampler = samplers.TPESampler(seed=seed)
    return search(
        param_space,
        objective,
        n_trials,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        direction=direction,
        directions=directions,
        timeout=timeout,
    )
