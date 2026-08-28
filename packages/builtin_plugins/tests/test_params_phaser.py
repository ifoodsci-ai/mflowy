"""annotated_params_phaser：Annotated 签名驱动的 YAML 值 → 类型实例转换（builtin 词汇的通用 phaser）。"""

from typing import Annotated

from mflowy.builtin_plugins.model.types import TASKTYPE
from mflowy.builtin_plugins.params_phaser import annotated_params_phaser
from mflowy.utils.study import ContinuousSpace, DiscreteSpace


def _phaser_for(fn):
    return annotated_params_phaser(fn)


def test_enum_value_and_name_forms():
    def h(task: Annotated[TASKTYPE | None, "任务类型"] = None, **_): ...

    conv = _phaser_for(h)
    assert conv({"task": "regressor"})["task"] is TASKTYPE.REGRESSION
    assert conv({"task": "CLASSIFICATION"})["task"] is TASKTYPE.CLASSIFICATION


def test_search_space_conversion():
    def h(
        lr: Annotated[float | ContinuousSpace[float] | None, "学习率"] = None,
        depth: Annotated[int | DiscreteSpace[int] | None, "深度"] = None,
        **_,
    ): ...

    conv = _phaser_for(h)
    out = conv({"lr": {"start": 0.001, "end": 0.1}, "depth": [3, 5, 7]})
    assert isinstance(out["lr"], ContinuousSpace)
    assert out["lr"].start == 0.001
    assert isinstance(out["depth"], DiscreteSpace)
    assert list(out["depth"]) == [3, 5, 7]


def test_non_annotated_passthrough():
    def h(df, lr: Annotated[float | None, "学习率"] = None, **_): ...

    conv = _phaser_for(h)
    out = conv({"df": " stays ", "lr": 0.5})
    assert out["df"] == " stays "  # 裸类型参数不转换（中间件注入域）
    assert out["lr"] == 0.5


def test_wired_builtin_modules_carry_phaser():
    """受影响模块已挂 phaser（x_y/model 族）——经 discover 链路可见"""
    from mflowy.driver.discover import _load_fn

    for step, module in [("X_y", "x_y"), ("model", "XGB"), ("model", "search_input")]:
        fn = _load_fn(step, module)
        assert fn is not None and callable(fn.convert_params), f"{step}.{module} 缺 params_phaser"
