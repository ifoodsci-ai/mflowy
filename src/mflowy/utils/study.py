"""Optuna 学习/搜索共享抽象。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

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


class DiscreteSpace[T: int | float | str](list[T]):
    def __init__(self, iterable=()):
        super().__init__(iterable)


type ParameterSearchSpace = ContinuousSpace | DiscreteSpace


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


def get_sampler(name: str = "tpe", **kwargs) -> BaseSampler:
    """按名称获取采样器类（lazy import optuna.samplers）"""
    from optuna import samplers

    _SAMPLER_FACTORIES = {
        "tpe": lambda: samplers.TPESampler(**samplers.TPESampler.hyperopt_parameters(), **kwargs),
    }
    try:
        return _SAMPLER_FACTORIES[name]()
    except KeyError as e:
        raise KeyError(f"未知的优化方法 '{name}'。目前支持: {list(_SAMPLER_FACTORIES.keys())}") from e


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
    """
    执行超参数搜索。

    Example:
        >>> space = {
        ...     "lr": ContinuousSpace(1e-5, 1e-1, step="log"),
        ...     "batch_size": DiscreteSpace([16, 32, 64]),
        ... }
        >>> def train(trial, lr, batch_size):
        ...     return lr * batch_size  # dummy
        >>> study = search(space, train, n_trials=50)
        >>> print(study.best_params)
    """

    def _objective(trial: Trial) -> T:
        params = suggest_params(trial, param_space)
        return objective(trial, **params)

    import optuna

    study = optuna.create_study(
        sampler=sampler or get_sampler(),
        pruner=pruner,
        direction=direction,
        directions=directions,
    )
    study.optimize(_objective, n_trials, timeout)
    return study
