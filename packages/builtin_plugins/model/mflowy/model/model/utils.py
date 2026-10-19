"""训练模块工具函数

提供 infer_task、merge_param_space、training 通用训练循环等。
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from mflowy.model.model.study import ContinuousSpace, DiscreteSpace, ParameterSearchSpace

from .types import TASKTYPE, MetricName

logger = logging.getLogger(__name__)


def validate_input(X: pd.DataFrame, y: pd.DataFrame) -> None:
    """验证输入数据（监督学习：X/y 恒成对）"""
    if len(X) != len(y):
        raise ValueError(f"X 和 y 长度不匹配: {len(X)} vs {len(y)}")


def infer_metric(task: TASKTYPE) -> MetricName:
    """根据任务类型推断默认评估指标"""
    if task == TASKTYPE.REGRESSION:
        return MetricName.MAE
    elif task == TASKTYPE.CLASSIFICATION:
        return MetricName.F1
    else:
        raise ValueError(f"未知任务类型: {task}")


def extract_search_spaces(
    hyper_params: dict[str, Any],
) -> tuple[dict[str, int | float | str], dict[str, ParameterSearchSpace]]:
    params = {}
    params_space = {}
    for name, p in hyper_params.items():
        if p is None:
            continue
        if isinstance(p, (ContinuousSpace, DiscreteSpace)):
            params_space[name] = p
        else:
            params[name] = p
    return params, params_space


# machinery 工厂跳过的 plumbing 形参（会话两动作不透传面；random_state 由 seed 专用参数归一）
_MACHINERY_SKIP = frozenset({"train_split", "x_preprocessors", "n_trials", "scoring", "random_state"})


def signature_params(fn) -> dict[str, Any]:
    """族节点签名的用户超参缺省值（单一真源 = 签名；plumbing 除外）。"""
    import inspect

    return {
        k: v.default
        for k, v in inspect.signature(fn).parameters.items()
        if k not in _MACHINERY_SKIP and v.default is not inspect.Signature.empty
    }


def make_machinery(fn, model_cls_of, fixed: dict | None = None, seed_key: str = "random_state", skip=()):
    """族节点 → 会话可直调的 estimator 机器（#62 两行为的共享底座）。

    缺省超参 = ``fn`` 签名默认值（单一真源）∪ ``fixed``（oob_score/verbose 等
    非签名机器参数）；``skip`` = 不进 params 的签名参数（如 LGBM 的
    early_stopping_rounds 走 fit_params）；调用面 ``(task, x_preprocessors,
    seed=None, **overrides) -> (make_pipe, base_params, param_space)``——
    overrides 标量 = pin、Space = 收窄。
    """
    fixed = dict(fixed or {})
    sig_defaults = {k: v for k, v in signature_params(fn).items() if k not in set(skip)}

    def machinery(task, x_preprocessors=None, seed=None, **overrides):
        from ._model import build_pipeline

        model_cls = model_cls_of()

        def make_pipe(**params):
            return build_pipeline(task, model_cls, x_preprocessors, params)

        hyper = {**sig_defaults, **fixed}
        hyper.update(overrides)
        if seed is not None:
            hyper[seed_key] = seed
        initial_params, param_space = extract_search_spaces(hyper)
        return make_pipe, initial_params, param_space

    return machinery


def space_to_json(space) -> dict:
    """Space → manifest/卡 JSON 形态（光标 h_space 与卡 hints 共用的持久化格式）。"""
    from mflowy.model.model.study import ContinuousSpace, DiscreteSpace

    if isinstance(space, ContinuousSpace):
        return {"kind": "continuous", "lo": space.start, "hi": space.end, "step": space.step}
    if isinstance(space, DiscreteSpace):
        return {"kind": "discrete", "values": list(space)}
    raise TypeError(f"未知空间类型 {type(space).__name__}")


def space_from_json(data: dict):
    """JSON 形态 → Space 本体（session 光标 h_space 的运行期重建）。"""
    from mflowy.model.model.study import ContinuousSpace, DiscreteSpace

    if data["kind"] == "continuous":
        return ContinuousSpace(data["lo"], data["hi"], data["step"])
    return DiscreteSpace(data["values"])


def normalize_space(param: str, declared, value):
    """用户空间声明（[lo,hi] / [lo,hi,step] / [v1,v2,…] / {lo,hi,step}）→ Space 本体。

    与签名声明的空间类型匹配（连续声明收数值界、离散声明收值清单），收窄越界
    相对声明空间校验（fail-loud 报两侧）；静态常量缺省走 pin 语义（ValueError 指路）。
    """
    from mflowy.model.model.study import ContinuousSpace, DiscreteSpace

    if isinstance(declared, ContinuousSpace):
        if isinstance(value, dict):
            spec = (value.get("lo"), value.get("hi"), value.get("step"))
        elif isinstance(value, (list, tuple)) and 2 <= len(value) <= 3:
            spec = (value[0], value[1], value[2] if len(value) == 3 else None)
        else:
            raise ValueError(f"空间声明[{param}] 须为 [lo, hi] / [lo, hi, step] / {{lo, hi, step}}，got {value!r}")
        lo, hi, step = spec
        if (
            isinstance(lo, bool)
            or isinstance(hi, bool)
            or not isinstance(lo, (int, float))
            or not isinstance(hi, (int, float))
            or not lo < hi
        ):
            raise ValueError(f"空间声明[{param}] 界须为数值且 lo < hi，got [{lo!r}, {hi!r}]")
        if lo < declared.start or hi > declared.end:
            raise ValueError(f"空间声明[{param}] 收窄 [{lo}, {hi}] 越出签名声明空间 [{declared.start}, {declared.end}]")
        return ContinuousSpace(lo, hi, step)
    if isinstance(declared, DiscreteSpace):
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"空间声明[{param}] 离散空间须为非空值清单（⊆ {list(declared)}），got {value!r}")
        unknown = [v for v in value if v not in declared]
        if unknown:
            raise ValueError(f"空间声明[{param}] 值 {unknown} 不在签名声明值域 {list(declared)}")
        return DiscreteSpace(list(value))
    raise ValueError(f"空间声明[{param}] 缺省是静态常量 {declared!r}——静态面走 pin，不可开空间")
