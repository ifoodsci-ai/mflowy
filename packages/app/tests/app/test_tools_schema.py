"""工具帮助面契约（#243）：``mflowy <tool> -h`` 即 API 文档——全部参数经 Annotated
Field 携带描述（CLI 旗标编译单源消费），工具描述带「何时使用」指南段。

原 MCP inputSchema 契约随 SDK 退役改由旗标编译面承载（同一真源 = pyfunc 签名）。
"""

import inspect
from importlib.metadata import entry_points
from typing import Optional, get_args, get_type_hints

from mflowy.app import cli
from pydantic import BaseModel
from pydantic.fields import FieldInfo

# 全部注册工具（mflowy.tools 组：平台 + 域静态声明；与 CLI 分发同一真源）
_TOOL_FNS = [ep.load() for ep in entry_points(group="mflowy.tools")]


def _help_text(fn) -> str:
    """格式化 help 去除换行/缩进——argparse 折行（含超长 token 断词）是版面
    工件，不构成「字段未揭示」。"""
    return "".join(cli.build_tool_parser(fn, prog=f"mflowy {fn.__name__}").format_help().split())


def _param_description(fn, name: str) -> str:
    hints = get_type_hints(fn, include_extras=True)
    return next((m.description for m in get_args(hints[name]) if isinstance(m, FieldInfo) and m.description), "")


def test_every_tool_param_has_schema_description():
    """每个参数的 Field description 必须非空（编译进 ``-h`` 的单源真源面）。"""
    missing = []
    for fn in _TOOL_FNS:
        hints = get_type_hints(fn, include_extras=True)
        params = [p for p in inspect.signature(fn).parameters if p in hints]
        assert params, f"{fn.__name__} 无参数（会话工具恒带寻址锚）"
        for name in params:
            if not _param_description(fn, name):
                missing.append(f"{fn.__name__}.{name}")
    assert missing == [], f"缺描述的参数: {missing}"


def test_every_tool_description_has_usage_guidelines():
    """工具描述 = Purpose 首段 + 「何时使用 <name>：」指南段（help 即 API 文档）。"""
    missing = [fn.__name__ for fn in _TOOL_FNS if f"何时使用 {fn.__name__}：" not in (inspect.getdoc(fn) or "")]
    assert missing == [], f"缺「何时使用」指南段的工具: {missing}"


def test_nested_prior_schema_revealed_in_help():
    """#233 验收线：``-h`` 须揭示 --prior 嵌套 schema（CleanPrior 字段面）——
    dict 注解由 Field description 自足承载（字段枚举进描述），编译进 help。"""
    from mflowy.data.tools import data_load_or_start

    text = _help_text(data_load_or_start)
    for key in ("target", "id_columns", "time_column", "expected_ranges", "invariants", "min_row_retention"):
        assert key in text, f"data_load_or_start -h 未揭示 CleanPrior 字段 {key}（嵌套 schema 通道空转）"
    assert "--prior" in text


def test_nested_cvcard_and_candidates_revealed_in_help():
    """#233 验收线：modeling_load_or_start 的 cv_card 键面与 candidates 形状可发现。"""
    from mflowy.model.tools import modeling_load_or_start

    text = _help_text(modeling_load_or_start)
    for key in ("cv", "test_size", "random_state"):
        assert key in text, f"modeling_load_or_start -h 未揭示 cv_card 键 {key}"
    assert "candidates" in text and "space_hints" in text


class Inner(BaseModel):
    """内层"""

    target: str
    weight: int = 1


class Card(BaseModel):
    """外层"""

    name: str
    inner: Inner


def test_pydantic_model_annotation_expands_nested_fields():
    """编译器对 pydantic 模型注解参数展开嵌套字段（模型注解形态的支持面——
    当前工具面为 dict+description 自足，模型注解经此通道同可达）。"""

    def tool_with_model(card: Optional[Card] = None):
        """探针。

        何时使用 tool_with_model：
        - 嵌套展开
        """
        return card

    text = cli.build_tool_parser(tool_with_model, prog="mflowy tool_with_model").format_help()
    assert "--card" in text
    assert "name" in text and "inner" in text  # 外层字段
    assert "target" in text and "weight" in text  # 内层字段递归展开
