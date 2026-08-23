"""_build_params_converter 纯注解驱动转换测试。

设计契约：类型完全由签名注解 + 值自身结构决定，无形状猜测：
- 连续空间注解 + dict 值 → ``ContinuousSpace(**val)``（dict 是 dataclass 自然构造形式）
- 离散空间注解 + list 值 → ``DiscreteSpace(val)``
- ``dict[K, ParameterSearchSpace]`` + dict 值 → 逐值 dict=连续 / list=离散 / 其他=warning
"""

from typing import Annotated

from mflowy.driver.handler import _build_params_converter
from mflowy.utils.study import ContinuousSpace, DiscreteSpace, ParameterSearchSpace


def _h_cont(x: Annotated[float | ContinuousSpace[float] | None, "连续"] = None):
    return x


def _h_disc(x: Annotated[str | DiscreteSpace[str] | None, "离散"] = None):
    return x


def _h_columns(x: Annotated[dict[str, ParameterSearchSpace] | None, "列约束"] = None):
    return x


class TestTopLevelContinuous:
    def test_dict_encoding(self):
        conv = _build_params_converter(_h_cont)
        out = conv({"x": {"start": 0.01, "end": 0.1, "step": "log"}})
        assert out["x"] == ContinuousSpace(0.01, 0.1, "log")

    def test_2field_dict_step_defaults_none(self):
        conv = _build_params_converter(_h_cont)
        out = conv({"x": {"start": 1, "end": 3}})
        assert out["x"] == ContinuousSpace(1, 3)

    def test_plain_scalar_untouched(self):
        conv = _build_params_converter(_h_cont)
        out = conv({"x": 0.1})
        assert out["x"] == 0.1

    def test_already_instance_skipped(self):
        conv = _build_params_converter(_h_cont)
        inst = ContinuousSpace(0.005, 0.3)
        out = conv({"x": inst})
        assert out["x"] is inst


class TestTopLevelDiscrete:
    def test_list_encoding(self):
        conv = _build_params_converter(_h_disc)
        out = conv({"x": ["sqrt", "log2"]})
        assert isinstance(out["x"], DiscreteSpace)
        assert list(out["x"]) == ["sqrt", "log2"]

    def test_plain_scalar_untouched(self):
        conv = _build_params_converter(_h_disc)
        out = conv({"x": "sqrt"})
        assert out["x"] == "sqrt"


class TestDictOfSpace:
    def test_mixed_dict_and_list_values(self):
        """columns: 连续列编码为 dict，离散列编码为 list，逐值按结构转换。"""
        conv = _build_params_converter(_h_columns)
        out = conv({"x": {"a": {"start": 1, "end": 3}, "b": [1, 2, 3]}})
        assert out["x"]["a"] == ContinuousSpace(1, 3)
        assert isinstance(out["x"]["b"], DiscreteSpace)
        assert list(out["x"]["b"]) == [1, 2, 3]

    def test_numeric_discrete_stays_discrete(self):
        """回归：离散数值列 [1, 2] / [1, 2, 3] 不再被形状猜测转成连续。"""
        conv = _build_params_converter(_h_columns)
        out = conv({"x": {"a": [1, 2], "b": [1, 2, 3]}})
        assert isinstance(out["x"]["a"], DiscreteSpace)
        assert list(out["x"]["a"]) == [1, 2]
        assert list(out["x"]["b"]) == [1, 2, 3]

    def test_unrecognized_value_warns_and_skips(self, caplog):
        conv = _build_params_converter(_h_columns)
        out = conv({"x": {"a": 42}})
        assert out["x"]["a"] == 42  # 非 dict/list → warning 跳过，原样保留
        assert any("需编码为 dict" in r.message for r in caplog.records)


class TestTaskTypeDualForm:
    """yaml 值（regression）与枚举名（REGRESSION）都应转换为 TASKTYPE"""

    def _conv(self):
        from typing import Annotated

        from mflowy.compute.model.types import TASKTYPE

        def _h(task: Annotated[TASKTYPE | None, "任务类型"] = None):
            return task

        return _build_params_converter(_h)

    def test_value_form(self):
        out = self._conv()({"task": "regressor"})
        from mflowy.compute.model.types import TASKTYPE

        assert out["task"] is TASKTYPE.REGRESSION

    def test_name_form(self):
        out = self._conv()({"task": "CLASSIFICATION"})
        from mflowy.compute.model.types import TASKTYPE

        assert out["task"] is TASKTYPE.CLASSIFICATION

    def test_invalid_raises_with_options(self):
        import pytest

        with pytest.raises(ValueError, match="TASKTYPE"):
            self._conv()({"task": "nope"})
