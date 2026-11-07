"""类型钩子：DiscreteSpace / DualFormStrEnum 的 pydantic 校验行为（validate_call 经同一套 schema 机器消费）。

- DiscreteSpace：yaml list 值 → 实例（pydantic 对 list 子类只做 is-instance 不构造，
  钩子在 list schema 之后构造——before 会被内层摊平成 plain list）
- DualFormStrEnum：值/名双形式（parse_enum 语义的钩子版，serializer._plain 写名、用户多用值）
"""

from typing import Annotated

import pytest
from mflowy.model.model.study import ContinuousSpace, DiscreteSpace, ParameterSearchSpace
from mflowy.model.model.types import TASKTYPE
from pydantic import TypeAdapter, ValidationError


class TestDiscreteSpaceHook:
    def test_yaml_list_to_instance(self):
        """承重场景：`T | DiscreteSpace[T]` 收 YAML list → 构造实例（无钩子则 union 全臂失败）"""
        ta = TypeAdapter(int | DiscreteSpace[int] | None)
        v = ta.validate_python([3, 5, 7])
        assert isinstance(v, DiscreteSpace)
        assert list(v) == [3, 5, 7]

    def test_scalar_arm_passthrough(self):
        ta = TypeAdapter(int | DiscreteSpace[int] | None)
        assert ta.validate_python(100) == 100

    def test_instance_idempotent(self):
        ta = TypeAdapter(int | DiscreteSpace[int] | None)
        v = ta.validate_python(DiscreteSpace([1, 2]))
        assert isinstance(v, DiscreteSpace) and list(v) == [1, 2]


class TestContinuousSpace:
    def test_dict_to_instance_native(self):
        """stdlib dataclass：dict→实例 pydantic 原生支持（无需钩子）"""
        ta = TypeAdapter(float | ContinuousSpace[float] | None)
        v = ta.validate_python({"start": 0.001, "end": 0.1})
        assert isinstance(v, ContinuousSpace)
        assert (v.start, v.end) == (0.001, 0.1)

    def test_log_step_literal(self):
        ta = TypeAdapter(float | ContinuousSpace[float] | None)
        assert ta.validate_python({"start": 1e-5, "end": 1e-1, "step": "log"}).step == "log"

    def test_nested_search_space_dict(self):
        """嵌套 dict[str, ParameterSearchSpace]：dict→连续 / list→离散逐 key 分派（原 phaser 手写循环）"""
        ta = TypeAdapter(dict[str, ParameterSearchSpace])
        out = ta.validate_python({"lr": {"start": 0.001, "end": 0.1}, "depth": [3, 5, 7]})
        assert isinstance(out["lr"], ContinuousSpace)
        assert isinstance(out["depth"], DiscreteSpace)


class TestDualFormEnum:
    def test_by_value(self):
        ta = TypeAdapter(Annotated[TASKTYPE | None, "任务类型"])
        assert ta.validate_python("regressor") is TASKTYPE.REGRESSION

    def test_by_name(self):
        ta = TypeAdapter(Annotated[TASKTYPE | None, "任务类型"])
        assert ta.validate_python("CLASSIFICATION") is TASKTYPE.CLASSIFICATION

    def test_instance_passthrough(self):
        ta = TypeAdapter(Annotated[TASKTYPE | None, "任务类型"])
        assert ta.validate_python(TASKTYPE.REGRESSION) is TASKTYPE.REGRESSION

    def test_invalid_raises(self):
        ta = TypeAdapter(Annotated[TASKTYPE | None, "任务类型"])
        with pytest.raises(ValidationError):
            ta.validate_python("nonexistent")
