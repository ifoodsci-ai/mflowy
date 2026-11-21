"""JSON 分发内核（run_tool）契约——CLI（旗标面）与 mflowy-job（容器面）的共用单源。

覆盖：未知/私有名拒绝、绑定错误（TypeError）与运行期错误分开报、输出契约
（查询类回显 / DAG 类流式上屏不回显）、JobOutcome/DAGResult 退出码映射、
observe 遥测观察缝。CLI 旗标层语义见 test_cli.py。
"""

from types import SimpleNamespace

from mflowy.app import runner


def _patch_eps(monkeypatch, **tools_map):
    """在 entry points 缝注入替身——测试替身钉真实消费面（ep.name / ep.load()）。"""
    import importlib.metadata

    eps = [SimpleNamespace(name=n, load=lambda fn=fn: fn) for n, fn in tools_map.items()]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: eps)


def test_unknown_tool_exits(monkeypatch, capsys):
    assert runner.run_tool("no-such-tool", {}) == 1
    assert "Unknown tool" in capsys.readouterr().err


def test_private_name_rejected(monkeypatch, capsys):
    _patch_eps(monkeypatch)  # 空注册表：表无名即不可达
    assert runner.run_tool("_private", {}) == 1
    assert "Unknown tool" in capsys.readouterr().err


def test_invalid_arguments_reported_as_invalid(monkeypatch, capsys):
    def tool(namespace=None):
        return []

    _patch_eps(monkeypatch, list_nodes=tool)
    assert runner.run_tool("list_nodes", {"unexpected": 1}) == 1
    assert "Invalid arguments" in capsys.readouterr().err


def test_runtime_error_not_masqueraded_as_invalid_args(monkeypatch, capsys):
    """工具体内抛的 TypeError 属运行期错误，不得报成 Invalid arguments。"""

    def boom(**kwargs):
        raise TypeError("runtime type blowup")

    _patch_eps(monkeypatch, boom_tool=boom)
    assert runner.run_tool("boom_tool", {}) == 1
    err = capsys.readouterr().err
    assert "Invalid arguments" not in err
    assert "runtime type blowup" in err


def test_async_tool_awaited(monkeypatch, capsys):
    async def probe() -> dict:
        return {"ok": True}

    _patch_eps(monkeypatch, probe=probe)
    assert runner.run_tool("probe", {}) == 0
    assert '"ok"' in capsys.readouterr().out


def test_info_tool_result_echoed(monkeypatch, capsys):
    """info/查询类工具的返回值即输出，回显 JSON。"""
    _patch_eps(monkeypatch, query_tool=lambda: [{"step": "load", "modules": ["csv"]}])
    assert runner.run_tool("query_tool", {}) == 0
    assert '"step": "load"' in capsys.readouterr().out


def test_workflow_result_not_echoed(monkeypatch, capsys):
    """建模类工具返回 DAGResult，输出由 DAG 实时 print 上屏，不重复回显。"""
    from mflowy.driver.dag import DAGResult

    _patch_eps(monkeypatch, modeling_tool=lambda: DAGResult("1", "finished", "", [], "graph"))
    assert runner.run_tool("modeling_tool", {}) == 0
    assert capsys.readouterr().out == ""


def test_duplicate_entry_point_names_fail_loud(monkeypatch, capsys):
    """mflowy.tools 组撞名（跨发行重名）→ 拒绝分发而非静默遮蔽（红线 1）。"""
    import importlib.metadata

    dup = SimpleNamespace(name="dup_tool", load=lambda: lambda: None)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [dup, dup])
    assert runner.run_tool("dup_tool", {}) == 1
    err = capsys.readouterr().err
    assert "撞名" in err
    assert "dup_tool" in err


def test_failed_dag_result_exits_nonzero(monkeypatch, capsys):
    """DAG 执行失败装在 DAGResult.status 正常返回——退出码必须反映（e2e/CI 断链防）。"""
    from mflowy.driver.dag import DAGResult

    _patch_eps(
        monkeypatch,
        modeling_tool=lambda **kw: DAGResult("1", "failed", "DAG aborted: [plot.x]failed: boom", [], "graph"),
    )
    assert runner.run_tool("modeling_tool", {}) == 1
    assert "DAG aborted" in capsys.readouterr().err


def test_job_outcome_failed_dag_exits_nonzero(monkeypatch, capsys):
    """JobOutcome 受理面（#156）：finished 内嵌 DAG 失败 → 退出码反映（容器 wrap 契约）。"""
    from mflowy.driver.dag import DAGResult
    from mflowy.model.job_provider.protocol import JobOutcome

    _patch_eps(
        monkeypatch,
        modeling_tool=lambda: JobOutcome.finished(dag_result=DAGResult("1", "failed", "boom", [], "g")),
    )
    assert runner.run_tool("modeling_tool", {}) == 1
    assert "boom" in capsys.readouterr().err


def test_job_outcome_finished_ok_exit_zero_no_echo(monkeypatch, capsys):
    """finished 内嵌成功 DAG → exit 0 且不重复回显（DAG 实时上屏约定延续）。"""
    from mflowy.driver.dag import DAGResult
    from mflowy.model.job_provider.protocol import JobOutcome

    _patch_eps(
        monkeypatch,
        modeling_tool=lambda: JobOutcome.finished(dag_result=DAGResult("1", "finished", "", [], "g")),
    )
    assert runner.run_tool("modeling_tool", {}) == 0
    assert capsys.readouterr().out == ""


class TestObserveSeam:
    """遥测观察缝：成功/失败两路回调；回调抛异常不打断分发（调用方自行兜底）。"""

    def test_observe_success_payload(self, monkeypatch):
        seen = []
        _patch_eps(monkeypatch, q=lambda: {"a": 1})
        rc = runner.run_tool("q", {}, observe=lambda n, kw, result, error: seen.append((n, result, error)))
        assert rc == 0
        assert seen == [("q", {"a": 1}, None)]

    def test_observe_error_payload(self, monkeypatch, capsys):
        seen = []

        def boom():
            raise ValueError("爆")

        _patch_eps(monkeypatch, boom=boom)
        rc = runner.run_tool("boom", {}, observe=lambda n, kw, result, error: seen.append((n, result, error)))
        assert rc == 1
        name, result, error = seen[0]
        assert name == "boom" and result is None and isinstance(error, ValueError)

    def test_no_observe_by_default(self, monkeypatch):
        """jobwrap（容器面）不传 observe——零遥测面，默认不回调。"""
        _patch_eps(monkeypatch, q=lambda: None)
        assert runner.run_tool("q", {}) == 0
