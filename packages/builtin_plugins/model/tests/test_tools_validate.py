"""validate_modeling_steps 失败契约 — 校验失败返回结构化 ``valid=False``，不抛异常。

官方 MCP SDK 对非 ToolError 异常脱敏：客户端只收到裸 ``Error executing tool <name>``，
失败原因留在服务端日志（crash 通道）。校验器的领域结果就是「不合法 + 原因」，
全部失败路径（缺文件 / 片段格式 / YAML 语法 / DAG 构建）走结构化返回，
MCP 客户端（agent）才能拿到详情自我纠正。
"""

import asyncio
import csv

from mcp.server.mcpserver import MCPServer
from mflowy.model import tools

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
      targets: ["y"]
"""


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_full_document_returns_structured_error(tmp_path):
    """传完整文档（顶层 mapping）是最自然的误用 → 返回自解释的片段格式提示，而非天书 ValueError"""
    f = _write(tmp_path / "doc.yaml", _FULL_DOCUMENT)
    result = tools.validate_modeling_steps(f)
    assert result["valid"] is False
    assert "步骤列表" in result["error"]
    assert "外壳" in result["error"]


def test_empty_file_returns_structured_error(tmp_path):
    f = _write(tmp_path / "empty.yaml", "")
    result = tools.validate_modeling_steps(f)
    assert result["valid"] is False
    assert "步骤列表" in result["error"]


def test_missing_file_returns_structured_error():
    result = tools.validate_modeling_steps("/no/such/steps.yaml")
    assert result["valid"] is False
    assert "文件不存在" in result["error"]


def test_yaml_syntax_error_returns_structured_error(tmp_path):
    """坏 YAML 的异常类型名+消息透传（原本被 SDK 脱敏）"""
    f = _write(tmp_path / "bad.yaml", "- name: [unclosed\n  bad: ::")
    result = tools.validate_modeling_steps(f)
    assert result["valid"] is False
    assert "Error" in result["error"]


def test_dag_build_failure_returns_structured_error(tmp_path):
    """片段格式正确但引用不存在的节点 → 构建期异常同样透传"""
    f = _write(tmp_path / "ghost.yaml", "- nope__ghost:\n    name: 幽灵\n")
    result = tools.validate_modeling_steps(f)
    assert result["valid"] is False
    assert "ghost" in result["error"]


def test_valid_fragment_passes(tmp_path):
    data = tmp_path / "data.csv"
    with open(data, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x1", "y"])
        for i in range(10):
            w.writerow([i, i * 2])
    f = _write(tmp_path / "steps.yaml", _FRAGMENT_TMPL.format(file_path=data))
    result = tools.validate_modeling_steps(f)
    assert result["valid"] is True
    assert "mermaid" in result["graph"]


def test_fragment_contract_documented_in_param_descriptions():
    """MCP schema 即 API 文档：片段格式契约写进所有吃 modeling_steps_yaml 的参数描述"""
    server = MCPServer(name="validate-desc-probe")
    for fn in (tools.validate_modeling_steps, tools.modeling, tools.explanation):
        server.tool()(fn)
    tool_list = asyncio.run(server.list_tools())
    by_name = {t.name: t for t in tool_list}
    for name in ("validate_modeling_steps", "modeling", "explanation"):
        desc = by_name[name].input_schema["properties"]["modeling_steps_yaml"]["description"]
        assert "步骤列表" in desc, f"{name} 参数描述未写明片段格式契约"
