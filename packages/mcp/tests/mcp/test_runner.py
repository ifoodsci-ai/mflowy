"""JSON runner（mflowy 入口）分发与错误语义测试。

runner 是 K8s Job 容器 / subprocess 的唯一入口：
- 工具名 hyphen→underscore 归一化
- 未知工具 / 私有名拒绝
- 参数绑定错误（TypeError）与运行期错误分开报
"""

from types import SimpleNamespace

import pytest
from mflowy.mcp import runner


def _patch_eps(monkeypatch, **tools_map):
    """在 entry points 缝注入替身——分发表塌缩后 runner 不再反射 tools 模块属性，
    测试替身钉真实消费面（ep.name / ep.load() 两消费点）。"""
    import importlib.metadata

    eps = [SimpleNamespace(name=n, load=lambda fn=fn: fn) for n, fn in tools_map.items()]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: eps)


def test_unknown_tool_exits(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["mflowy", "no-such-tool"])
    with pytest.raises(SystemExit):
        runner.main()
    assert "Unknown tool" in capsys.readouterr().err


def test_private_name_rejected(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["cmd", "_private"])
    with pytest.raises(SystemExit):
        runner.main()
    assert "Unknown tool" in capsys.readouterr().err


def test_hyphen_normalized_to_underscore(monkeypatch):
    """连字符工具名应归一化后命中 pyfunc（如 some-tool → some_tool）"""
    sentinel = object()
    monkeypatch.setattr("sys.argv", ["cmd", "some-tool"])
    _patch_eps(monkeypatch, some_tool=lambda: sentinel)
    runner.main()  # 不抛 SystemExit 即命中


def test_invalid_arguments_reported_as_invalid(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["cmd", "list_nodes", '{"unexpected": 1}'])
    with pytest.raises(SystemExit) as ei:
        runner.main()
    assert ei.value.code == 1
    assert "Invalid arguments" in capsys.readouterr().err


def test_runtime_error_not_masqueraded_as_invalid_args(monkeypatch, capsys):
    """工具体内抛的 TypeError 属运行期错误，不得报成 Invalid arguments"""

    def boom(**kwargs):
        raise TypeError("runtime type blowup")

    monkeypatch.setattr("sys.argv", ["cmd", "boom_tool"])
    _patch_eps(monkeypatch, boom_tool=boom)
    with pytest.raises(SystemExit):
        runner.main()
    err = capsys.readouterr().err
    assert "Invalid arguments" not in err
    assert "runtime type blowup" in err


def test_non_json_args_friendly_error(monkeypatch, capsys):
    """非 JSON 参数（如裸位置参数 load）报友好用法而非裸 traceback，exit 2"""
    monkeypatch.setattr("sys.argv", ["cmd", "list_nodes", "load"])
    with pytest.raises(SystemExit) as ei:
        runner.main()
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "Invalid JSON arguments" in err
    assert "cmd list_nodes" in err  # 用法示例


def test_info_tool_result_echoed(monkeypatch, capsys):
    """info/查询类工具的返回值即输出，回显 JSON（cmd 作为三入口之一可独立使用）"""
    monkeypatch.setattr("sys.argv", ["cmd", "query_tool"])
    _patch_eps(monkeypatch, query_tool=lambda: [{"step": "load", "modules": ["csv"]}])
    runner.main()
    out = capsys.readouterr().out
    assert '"step": "load"' in out


def test_workflow_result_not_echoed(monkeypatch, capsys):
    """建模类工具返回 DAGResult，输出由 DAG 实时 print 上屏，不重复回显"""
    from mflowy.driver.dag import DAGResult

    monkeypatch.setattr("sys.argv", ["cmd", "modeling_tool"])
    _patch_eps(monkeypatch, modeling_tool=lambda: DAGResult("e", "1", "d", "finished", "", [], "graph"))
    runner.main()
    assert capsys.readouterr().out == ""


def test_duplicate_entry_point_names_fail_loud(monkeypatch, capsys):
    """mflowy.tools 组撞名（跨发行重名）→ 拒绝分发而非静默遮蔽（红线 1）。"""
    import importlib.metadata

    dup = SimpleNamespace(name="dup_tool", load=lambda: lambda: None)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [dup, dup])
    monkeypatch.setattr("sys.argv", ["cmd", "dup_tool"])
    with pytest.raises(SystemExit) as ei:
        runner.main()
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "撞名" in err
    assert "dup_tool" in err


def test_json_error_hint_uses_node_vocabulary(monkeypatch, capsys):
    """JSON 解析错误的示例提示用 node 语境词汇（ns）——step 词汇已退役"""
    monkeypatch.setattr("sys.argv", ["cmd", "list_nodes", "not-json"])
    with pytest.raises(SystemExit) as ei:
        runner.main()
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert '"ns"' in err
    assert "step" not in err


def test_non_mapping_json_args_get_object_hint(monkeypatch, capsys):
    """非 JSON 对象实参（如裸 int）报对象格式提示，而非裸的 bind TypeError"""
    monkeypatch.setattr("sys.argv", ["cmd", "get_node_info", "123"])
    with pytest.raises(SystemExit) as ei:
        runner.main()
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert '"ns"' in err
    assert "参数须为 JSON 对象" in err


def test_failed_dag_result_exits_nonzero(monkeypatch, capsys):
    """DAG 执行失败装在 DAGResult.status 正常返回——runner 须转非零退出码。

    防 e2e/CI 断链（prior bug：SHAP 5/8 中止而 make e2e exit=0，失败被吞）。
    """
    import mflowy.model.tools as model_tools
    from mflowy.driver.dag import DAGResult

    monkeypatch.setattr("sys.argv", ["cmd", "modeling", "{}"])
    monkeypatch.setattr(
        model_tools,
        "modeling",
        lambda **kw: DAGResult("e", "1", "d", "failed", "DAG aborted: [plot.x]failed: boom", [], "g"),
    )
    with pytest.raises(SystemExit) as ei:
        runner.main()
    assert ei.value.code == 1
    assert "DAG aborted" in capsys.readouterr().err
