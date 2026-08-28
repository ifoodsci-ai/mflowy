"""插件侧参数转换器（params_phaser）：YAML 原始值 → 签名类型实例。

driver 内核对数据结构零感知（@handler 的 ``params_phaser`` 由插件侧自注册）——
本模块是 builtin 词汇的通用 phaser：内省 ``Annotated[T, "描述"]`` 签名，
转换 union 中的 Enum（值/名双形式）与 model.study 的搜索空间类型。
第三方插件带自有数据结构时，写自己的 phaser 挂进 @handler 即可。
"""

import logging
from enum import Enum
from typing import Annotated, get_args, get_origin, get_type_hints

from mflowy.builtin_plugins.model.study import ContinuousSpace, DiscreteSpace, ParameterSearchSpace

logger = logging.getLogger(__name__)


def annotated_params_phaser(fn) -> "callable":
    """从 handler 函数签名构建参数转换器：``Annotated`` 参数按类型转换，其余透传。"""
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        return lambda params: params

    param_types: dict[str, object] = {}
    for name, hint in hints.items():
        if get_origin(hint) is Annotated:
            param_types[name] = get_args(hint)[0]

    def convert(params: dict) -> dict:
        def _convert(val, typ):
            args = set(get_args(typ) or (typ,))
            # union 中任意 Enum 子类：字符串值/名互转
            enum_types = [a for a in args if isinstance(a, type) and issubclass(a, Enum)]
            if enum_types and isinstance(val, str) and not isinstance(val, Enum):
                from mflowy.driver.config import parse_enum

                return parse_enum(enum_types[0], val)
            if {ContinuousSpace, ContinuousSpace[float], ContinuousSpace[int]} & args and isinstance(val, dict):
                val = ContinuousSpace(**val)
            elif {DiscreteSpace, DiscreteSpace[int], DiscreteSpace[float], DiscreteSpace[str]} & args and isinstance(
                val, list
            ):
                val = DiscreteSpace(val)
            elif dict[str, ParameterSearchSpace] in args and isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, dict):
                        val[k] = ContinuousSpace(**v)
                    elif isinstance(v, list):
                        val[k] = DiscreteSpace(v)
                    else:
                        logger.warning(f"列约束 '{k}' 需编码为 dict(连续) 或 list(离散)，got {v!r}，跳过转换")
            return val

        for name, typ in param_types.items():
            if name not in params:
                continue
            params[name] = _convert(params[name], typ)
        return params

    return convert
