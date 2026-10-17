"""driver 内核契约：@handler 内嵌 validate_call 的运行时参数解析与校验。

- 时机：节点执行期（fail-late）——NodeConf.params 保持原始 dict，解析期不转换
- 范围：全签名校验；裸类型/无注解的注入参数（第三方任意类型）按 Any/is-instance 直通
- 约定：Annotated[T, "描述"] 旧式字符串 metadata 被忽略按 T 校验；约束走 Field；
  自定义数据结构的转换挂类型的 __get_pydantic_core_schema__（词汇主人侧）
"""

from enum import StrEnum
from typing import Annotated

import pytest
from mflowy.driver.config import NodeConf
from mflowy.driver.context import Context
from mflowy.driver.handler import handler
from pydantic import Field, ValidationError


class Color(StrEnum):
    RED = "red"
    BLUE = "blue"


def _ctx(**params):
    return Context(NodeConf(name="under_test", node_key="load__under_test", params=dict(params)))


@handler()
def _plain(x: Annotated[int, "倍数"] = 1, **_):
    return x


@handler()
def _constrained(x: Annotated[int, Field(ge=0, le=10, description="0~10 的整数")], **_):
    return x


@handler()
def _takes_arbitrary(payload, x: Annotated[int, "倍数"] = 1, **_):
    """payload 无注解：模拟参数边绑定期注入的任意类型对象（迭代器/DataFrame）"""
    return type(payload).__name__, x


class TestRuntimeValidation:
    def test_dispatch_coerces_yaml_scalar(self, fake_tracking):
        """调度链（.handler 路径）上 YAML 标量按签名类型转换（str→int，lax）"""
        assert _plain.handler(_ctx(x="7")) == 7

    def test_params_stay_raw_on_conf(self, fake_tracking):
        """解析期不转换：NodeConf.params 原样保留原始 YAML 值"""
        conf = NodeConf(name="plain", params={"x": "raw-dict"}, node_key="load__plain")
        assert conf.params == {"x": "raw-dict"}

    def test_field_constraint_rejected_at_dispatch(self, fake_tracking):
        with pytest.raises(ValidationError, match="x"):
            _constrained.handler(_ctx(x=11))

    def test_string_annotated_convention_validates(self, fake_tracking):
        """旧式 Annotated[T, "描述"]：字符串 metadata 被忽略，按 T 校验"""
        assert _plain.handler(_ctx(x="3")) == 3

    def test_enum_by_value_at_dispatch(self, fake_tracking):
        @handler()
        def _colored(color: Annotated[Color | None, "颜色"] = None, **_):
            return color

        assert _colored.handler(_ctx(color="red")) is Color.RED

    def test_defaults_not_validated(self, fake_tracking):
        """未出现在 YAML params 的参数走函数签名默认值，不经校验"""
        assert _plain.handler(_ctx()) == 1


class TestInjectionPassthrough:
    def test_unannotated_injected_object_passthrough(self, fake_tracking):
        """无注解参数接收任意类型对象（参数边绑定.kwargs 投递，同 _bind 入口）按 is-instance 直通"""
        kind, x = _takes_arbitrary.handler(_ctx(x="5"), payload=object())
        assert kind == "object"
        assert x == 5


class TestDispatchMarker:
    def test_handler_chain_marker_attached(self, fake_tracking):
        """.handler 调度链属性挂载（discover 单属性 marker 的内核侧依据）"""
        assert callable(_plain.handler)
        assert _plain.node_key == "_plain"  # 身份属性直挂原函数；私有前导保留（单一来源 config.node_key）
