"""Optuna 学习/搜索共享抽象。"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from contextlib import contextmanager
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
    def __get_pydantic_core_schema__(cls, source_type, node):
        """yaml list 值 → DiscreteSpace 实例。

        pydantic 对 list 子类只做 is-instance 不构造（LIST_TYPES 精确匹配不含子类），
        故在 list schema 校验之后包一层构造——after 而非 before：before 构造的实例
        会被内层 list schema 重新摊平成 plain list。
        """
        args = get_args(source_type)
        return core_schema.no_info_after_validator_function(cls, node(list[args[0]] if args else list))


type ParameterSearchSpace = ContinuousSpace | DiscreteSpace

# ── study 落库锚（#244 零拷贝单源化） ──────────────────────────────────────
# 链上恒名三源（study 名 / 工件名 / 留痕 tag）：名随文件词干漂移会让续跑
# load_if_exists 落空（#158 事后剖）。会话域内直连路径（_open_study，
# study_name = 结构键）不经此锚——两路径命名约定各自自洽。
STUDY_NAME = "study"
STUDY_ARTIFACT_FILE = "study.db"
STUDY_TRACE_TAG = "mflowy.study_db"

# 显式锚（#89 ①修正）：非空时 search() 建 study 走该 sqlite + load_if_exists——
# 同一路径重复搜索即增量续跑（复用前史）；零拷贝续跑锚的即前史 run 工件树内本体。
_study_storage: ContextVar[str | None] = ContextVar("mflowy_study_storage", default=None)
# run 作用域活写开关（#158 §3 → #244 语义固化）：hyperparameters_search 流程置位
# ——无显式锚时 study 直驻 **active run 工件根**（位置对齐 log_artifact，zip 往返
# 带出、失败 run 也有活文件兜底；搜索终 tag 相对路径留痕，零拷贝）
_study_live: ContextVar[bool] = ContextVar("mflowy_study_live", default=False)
# 缺件回退的下载暂存种子（job 级 resume 通道）：search() live 锚落家时搬入本 run
# 工件根（一次搬运即正典居所），消费即清——单 study 契约（多 model 步续跑须节点
# 参数 inherit_study_run_id 各自寻址）
_study_seed: ContextVar[str | None] = ContextVar("mflowy_study_seed", default=None)
# 本次 search 实际落库路径（供搜索尾 tag 留痕判定——单线程 DAG 内「search() →
# 留痕」紧邻，无交错）
_study_db_path_used: ContextVar[str | None] = ContextVar("mflowy_study_db_path_used", default=None)


def set_study_storage(path: str | None) -> Token[str | None]:
    """锚定 search() 的 study 落库位置（显式路径——零拷贝续跑锚前史 run 工件树内
    本体；job_provider 线程内 set，用毕 reset——token 语义同 set_task_dir）。"""
    return _study_storage.set(path)


def reset_study_storage(token: Token[str | None]) -> None:
    _study_storage.reset(token)


def set_study_live(flag: bool) -> Token[bool]:
    """置位 run 作用域活写（无显式锚时 search() 直驻 active run 工件根）。"""
    return _study_live.set(flag)


def reset_study_live(token: Token[bool]) -> None:
    _study_live.reset(token)


def set_study_seed(path: str | None) -> Token[str | None]:
    """置位下载暂存种子（缺件回退路径——恢复树缺件时 ``download_run_artifact``
    取回的 study.db；search() live 锚落家时搬入，消费即清）。"""
    return _study_seed.set(path)


def reset_study_seed(token: Token[str | None]) -> None:
    _study_seed.reset(token)


def study_db_path_used() -> str | None:
    """最近一次 search() 的实际落库 sqlite 路径（未落库/内存态 = None）。"""
    return _study_db_path_used.get()


def assert_inherit_search_face(inherit_study_run_id: str | None, param_space: dict) -> None:
    """节点面守卫（单源）：inherit_study_run_id 仅 search 行为（参数含搜索空间）可用
    ——全 pin（training 行为）同传 = 语义误用 fail-loud。"""
    if inherit_study_run_id is not None and not param_space:
        raise ValueError("inherit_study_run_id 仅 search 行为（参数含搜索空间）可用——全 pin = training 行为无 study")


def locate_study(run_id: str) -> tuple[str | None, str]:
    """run 寻址定位前史 study 工件（#244 零拷贝链解析面）：tag（``STUDY_TRACE_TAG``）
    相对路径优先、工件清单回退（#199 存量 run）；返回 ``(本地直驻路径 | None, 相对路径名)``。

    - 本地工件树（file scheme / 裸路径根）且文件在 → 直驻路径（零拷贝增量）
    - None = 非本地根/缺件——调用方走 download 回退（搬入本 run 工件根落家）
    - tag 与工件清单皆缺 → fail-loud 附清单（仅可续跑 hyperparameters_search 产出的 run）
    """
    from mflowy.utils.mlflow import local_artifact_root, run_artifacts, run_tags_and_artifact_uri

    arts = run_artifacts(run_id)
    tags, artifact_uri = run_tags_and_artifact_uri(run_id)
    rel = tags.get(STUDY_TRACE_TAG) or next((a for a in arts if a == STUDY_ARTIFACT_FILE), None)
    if rel is None:
        raise ValueError(
            f"study 续跑源 run {run_id!r} 无 study 工件（tag {STUDY_TRACE_TAG} 与工件清单均缺，"
            f"现有工件: {arts or '无'}）——仅可续跑 hyperparameters_search 产出的 run"
        )
    root = local_artifact_root(artifact_uri)
    if root is not None and (candidate := root / rel).is_file():
        return str(candidate), rel
    return None, rel


@contextmanager
def inherited_study(run_id: str):
    """节点级前史 study 锚定（#244，模型步 run 窗口内）：定位前史 study——本地工件
    树在 → 零拷贝直锚（前史 run 树内增量，搜索终不拷回不留痕——study 单源居前史
    run 树）；缺件/非本地根 → download 回退搬入本 run 工件根落家（一次搬运即
    正典居所）。"""
    from mflowy.utils.mlflow import active_artifact_root, download_run_artifact

    local, rel = locate_study(run_id)
    if local is None:
        root = active_artifact_root(local_only=True)
        if root is None:
            raise RuntimeError("缺件回退须在模型步 run 窗口内落家（无 active run 即无居所）")
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(download_run_artifact(run_id, rel), dest)
        local = str(dest)
    token = set_study_storage(local)
    try:
        yield
    finally:
        reset_study_storage(token)


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

    path = _study_storage.get()
    if path is None and _study_live.get():
        # run 作用域活写（#158 §3 → #244 语义固化）：直驻 active run 工件根（位置
        # 对齐 log_artifact——工件树即正典居所，零拷贝；失败 run 也有活文件兜底）；
        # 非本地工件根 fail-loud（sqlite 直写只对本地树成立）。下载暂存种子（缺件
        # 回退）在此搬入落家，消费即清
        from mflowy.utils.mlflow import active_artifact_root

        if (root := active_artifact_root(local_only=True)) is not None:
            path = str(root / STUDY_ARTIFACT_FILE)
            if (seed := _study_seed.get()) is not None:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(seed, path)
                _study_seed.set(None)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # 落库锚生效（#89 ①修正 / #158 续跑）：sqlite + load_if_exists——同库重复
        # 搜索复用前史（增量续跑）。study 名 = 链上恒名（名随文件词干漂移会让续跑
        # load_if_exists 落空）
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=f"sqlite:///{path}",
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
            directions=directions,
        )
        _study_db_path_used.set(path)
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
