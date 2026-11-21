"""validate_modeling_steps 失败契约 — 校验失败返回结构化 ``valid=False``，不抛异常。

校验失败若走异常通道，CLI 分发只会回裸 ``Error: ...`` 退出码。校验器的领域
结果就是「不合法 + 原因」，全部失败路径（片段格式 / YAML 语法 / DAG 构建）
走结构化返回，调用方（agent）才能拿到详情自我纠正。

入参为 yaml **明文**（非文件路径——CLI 边界不假设与引擎共享文件系统语义）。
"""

import csv
from typing import get_args, get_type_hints

import pytest
from mflowy.model import tools
from pydantic.fields import FieldInfo


def _param_description(fn, param: str) -> str:
    """Annotated Field description 直读（签名即 schema 的单源探针——CLI 旗标
    help 编译消费同一面，域包零 SDK/零 app import）。"""
    hints = get_type_hints(fn, include_extras=True)
    for meta in get_args(hints[param]):
        if isinstance(meta, FieldInfo) and meta.description:
            return meta.description
    return ""


_FULL_DOCUMENT = """\
name: 完整文档
description: 带 name/description/flow 外壳的完整 YAML（误用示范）
flow:
  - name: 导入
    namespace: load
    node_name: file
    params:
      file_path: /tmp/never.csv
"""

_FRAGMENT_TMPL = """\
- load__file:
    name: 导入
    params:
      file_path: {file_path}

- x_y:
    name: 划分目标
    params:
      target: y
"""


def test_full_document_returns_structured_error():
    """传完整文档（顶层 mapping）是最自然的误用 → 返回自解释的片段格式提示，而非天书 ValueError"""
    result = tools.validate_modeling_steps(_FULL_DOCUMENT)
    assert result["valid"] is False
    assert "步骤列表" in result["error"]
    assert "外壳" in result["error"]


def test_empty_text_returns_structured_error():
    result = tools.validate_modeling_steps("")
    assert result["valid"] is False
    assert "步骤列表" in result["error"]


def test_yaml_syntax_error_returns_structured_error():
    """坏 YAML 的异常类型名+消息透传（原本被 SDK 脱敏）"""
    result = tools.validate_modeling_steps("- name: [unclosed\n  bad: ::")
    assert result["valid"] is False
    assert "Error" in result["error"]


def test_dag_build_failure_returns_structured_error():
    """片段格式正确但引用不存在的节点 → 构建期异常同样透传"""
    result = tools.validate_modeling_steps("- nope__ghost:\n    name: 幽灵\n")
    assert result["valid"] is False
    assert "ghost" in result["error"]


def test_valid_fragment_passes(tmp_path):
    data = tmp_path / "data.csv"
    with open(data, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x1", "y"])
        for i in range(10):
            w.writerow([i, i * 2])
    result = tools.validate_modeling_steps(_FRAGMENT_TMPL.format(file_path=data))
    assert result["valid"] is True
    assert "mermaid" in result["graph"]


def test_fragment_contract_documented_in_param_descriptions():
    """签名即 API 文档（CLI help 编译同一真源）：片段格式契约写进所有吃 modeling_steps 的参数描述"""
    for name in ("validate_modeling_steps", "modeling", "explanation"):
        desc = _param_description(getattr(tools, name), "modeling_steps")
        assert desc, f"{name}.modeling_steps 参数缺 Field description"
        assert "步骤列表" in desc, f"{name} 参数描述未写明片段格式契约"


def test_fragment_gate_single_predicate():
    """#83 单源红线：片段门谓词住 templates.parse_steps_fragment——预检面与提交侧同一拒绝集。"""
    from mflowy.model.job_provider.local import _require_steps
    from mflowy.model.templates import parse_steps_fragment

    for bad in ["", "   ", _FULL_DOCUMENT, "42"]:
        with pytest.raises(ValueError, match="步骤列表片段"):
            parse_steps_fragment(bad)
        assert tools.validate_modeling_steps(bad)["valid"] is False
        with pytest.raises(ValueError, match="步骤列表片段"):
            _require_steps(bad)
