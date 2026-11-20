"""CLI 契约测试（#243）：旗标编译 / pydantic 校验单源 / 元旗标剥离 / 帮助面 / 分发集成。

探针工具经 entry points 缝注入（与 runner 测试同一惯用法）——CLI 消费面 =
``ep.name`` / ``ep.value``（分组判定）/ ``ep.load()`` 三点，替身钉真实契约。
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
from types import SimpleNamespace
from typing import Annotated

import pytest
from mflowy.app import cli
from pydantic import Field


def _patch_eps(monkeypatch, *extra, **tools_map):
    """entry points 缝注入：kwargs 形 = 模块前缀缺省（不参与分组断言的普通替身）；
    extra 形 = SimpleNamespace 列表（带 value 模块路径，供分组判定）。"""
    eps = list(extra) + [
        SimpleNamespace(name=n, value=f"mflowy.app.tools:{n}", load=lambda fn=fn: fn) for n, fn in tools_map.items()
    ]
    monkeypatch.setattr(importlib_metadata, "entry_points", lambda group: eps)


def _probe_tool(
    name_s: Annotated[str, Field(description="名字")] = "x",
    count: Annotated[int, Field(description="数量")] = 3,
    ratio: Annotated[float, Field(description="比率")] = 0.5,
    verbose: Annotated[bool, Field(description="详细输出")] = False,
    tags: Annotated[list[str] | None, Field(description="标签清单")] = None,
    prior: Annotated[dict | None, Field(description="先验卡")] = None,
):
    """探针工具——六型参数各一。

    何时使用 probe_tool：
    - 测试旗标编译
    - 测试缺省值下传
    """
    return {"name_s": name_s, "count": count, "ratio": ratio, "verbose": verbose, "tags": tags, "prior": prior}


class TestFlagCompilation:
    def test_kebab_flag_names(self):
        parser = cli.build_tool_parser(_probe_tool, prog="mflowy probe_tool")
        names = {opt for action in parser._actions for opt in action.option_strings}
        assert {
            "--name-s",
            "--count",
            "--ratio",
            "--verbose",
            "--no-verbose",
            "--tags",
            "--prior",
            "--kwargs-json",
        } <= names

    def test_scalar_typed_and_default_applied(self, monkeypatch, capsys):
        """标量旗标类型转换；未传旗标 = 函数签名缺省生效（SUPPRESS 面）。"""
        seen = {}

        def tool(a: Annotated[int, Field(description="甲")] = 1, b: Annotated[str, Field(description="乙")] = "d"):
            seen.update(a=a, b=b)
            return "ok"

        _patch_eps(monkeypatch, t=tool)
        monkeypatch.setattr("sys.argv", ["mflowy", "t", "--a", "5"])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 0
        assert seen == {"a": 5, "b": "d"}

    def test_bool_optional_action_both_forms(self, monkeypatch):
        seen = {}

        def tool(flag: Annotated[bool, Field(description="开关")] = False):
            seen["flag"] = flag
            return 1

        _patch_eps(monkeypatch, t=tool)
        for argv, expected in (
            (["mflowy", "t", "--flag"], True),
            (["mflowy", "t", "--no-flag"], False),
            (["mflowy", "t"], False),
        ):
            monkeypatch.setattr("sys.argv", argv)
            with pytest.raises(SystemExit) as ei:
                cli.main()
            assert ei.value.code == 0
            assert seen["flag"] is expected

    def test_json_flag_decodes_list_and_dict(self, monkeypatch):
        seen = {}

        def tool(
            tags: Annotated[list[str] | None, Field(description="标签")] = None,
            prior: Annotated[dict | None, Field(description="先验")] = None,
        ):
            seen.update(tags=tags, prior=prior)
            return []

        _patch_eps(monkeypatch, t=tool)
        monkeypatch.setattr("sys.argv", ["mflowy", "t", "--tags", '["a","b"]', "--prior", '{"target": "y"}'])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 0
        assert seen == {"tags": ["a", "b"], "prior": {"target": "y"}}

    def test_json_flag_invalid_json_usage_error(self, monkeypatch, capsys):
        _patch_eps(monkeypatch, probe_tool=_probe_tool)
        monkeypatch.setattr("sys.argv", ["mflowy", "probe_tool", "--prior", "{not-json"])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 2
        assert "非合法 JSON" in capsys.readouterr().err

    def test_required_flag_missing_usage_error(self, monkeypatch, capsys):
        def tool(must: Annotated[str, Field(description="必填项")]):
            return must

        _patch_eps(monkeypatch, t=tool)
        monkeypatch.setattr("sys.argv", ["mflowy", "t"])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 2
        assert "--must" in capsys.readouterr().err

    def test_field_description_lands_in_help(self):
        parser = cli.build_tool_parser(_probe_tool, prog="mflowy probe_tool")
        text = parser.format_help()
        assert "先验卡" in text and "数量" in text  # Field description → help


class TestKwargsJsonEscapeHatch:
    def test_full_payload_passes(self, monkeypatch, capsys):
        seen = {}

        def tool(
            a: Annotated[int, Field(description="甲")] = 1, b: Annotated[str | None, Field(description="乙")] = None
        ):
            seen.update(a=a, b=b)
            return {"echo": True}

        _patch_eps(monkeypatch, t=tool)
        monkeypatch.setattr("sys.argv", ["mflowy", "t", "--kwargs-json", '{"a": 7, "b": "z"}'])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 0
        assert seen == {"a": 7, "b": "z"}
        assert json.loads(capsys.readouterr().out)["echo"] is True

    def test_flag_and_json_same_key_rejected(self, monkeypatch, capsys):
        """参数来源冲突（红线 1）：具名旗标与 --kwargs-json 同键拒绝，绝不静默覆写。"""
        _patch_eps(monkeypatch, probe_tool=_probe_tool)
        monkeypatch.setattr("sys.argv", ["mflowy", "probe_tool", "--count", "2", "--kwargs-json", '{"count": 9}'])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 2
        assert "参数来源冲突" in capsys.readouterr().err

    def test_json_payload_not_object_rejected(self, monkeypatch, capsys):
        _patch_eps(monkeypatch, probe_tool=_probe_tool)
        monkeypatch.setattr("sys.argv", ["mflowy", "probe_tool", "--kwargs-json", "[1,2]"])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 2
        assert "JSON 对象" in capsys.readouterr().err


class TestValidationSingleSource:
    def test_unknown_key_fail_loud(self, monkeypatch, capsys):
        """未知键经 pydantic（extra=forbid）拒绝——旗标层零校验，运行时单源。"""
        _patch_eps(monkeypatch, probe_tool=_probe_tool)
        monkeypatch.setattr("sys.argv", ["mflowy", "probe_tool", "--kwargs-json", '{"no_such": 1}'])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 2
        assert "no_such" in capsys.readouterr().err

    def test_pydantic_coerce_string_int(self, monkeypatch):
        """--kwargs-json 里的字符串 "15" 经 pydantic coerce 为 int（运行时校验单源）。"""
        seen = {}

        def tool(n: Annotated[int, Field(description="数")] = 0):
            seen["n"] = n
            return n

        _patch_eps(monkeypatch, t=tool)
        monkeypatch.setattr("sys.argv", ["mflowy", "t", "--kwargs-json", '{"n": "15"}'])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 0
        assert seen["n"] == 15

    def test_type_violation_usage_error(self, monkeypatch, capsys):
        _patch_eps(monkeypatch, probe_tool=_probe_tool)
        monkeypatch.setattr("sys.argv", ["mflowy", "probe_tool", "--kwargs-json", '{"count": "abc"}'])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 2
        assert "count" in capsys.readouterr().err


class TestTelemetryMetaFlags:
    def test_meta_flags_stripped_and_injected_into_observe(self, monkeypatch):
        """遥测元旗标 CLI 层剥离——不进 pyfunc 签名、随 observe 缝进遥测。"""
        captured = {}

        import mflowy.app.runner as runner_mod

        def _fake_run_tool(name, kwargs, *, observe=None):
            captured.update(name=name, kwargs=kwargs)
            if observe:
                observe(name, kwargs, None, None)  # 模拟成功调用回调
            return 0

        monkeypatch.setattr(runner_mod, "run_tool", _fake_run_tool)
        reported = {}
        import mflowy.app.telemetry as telemetry_mod

        monkeypatch.setattr(
            telemetry_mod,
            "report_call",
            lambda tool, kw, result=None, error=None, session_id=None, intent=None: reported.update(
                session_id=session_id, intent=intent
            ),
        )
        monkeypatch.setattr(
            "sys.argv", ["mflowy", "t", "--telemetry-session-id", "sess-1", "--telemetry-intent", "清洗数据"]
        )

        def tool(a: Annotated[int, Field(description="甲")] = 1):
            return a  # pragma: no cover —— run_tool 已被替身接住

        _patch_eps(monkeypatch, t=tool)
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 0
        assert captured["kwargs"] == {"a": 1}  # 元旗标已被剥离
        assert reported == {"session_id": "sess-1", "intent": "清洗数据"}


class TestDispatchIntegration:
    def test_hyphen_tool_name_normalized(self, monkeypatch, capsys):
        """连字符工具名归一化（some-tool → some_tool，与旧 runner 惯例一致）。"""
        _patch_eps(monkeypatch, some_tool=_probe_tool)
        monkeypatch.setattr("sys.argv", ["mflowy", "some-tool"])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 0

    def test_unknown_tool_fail_loud_with_candidates(self, monkeypatch, capsys):
        _patch_eps(monkeypatch, alpha=_probe_tool, beta=_probe_tool)
        monkeypatch.setattr("sys.argv", ["mflowy", "no-such"])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 1
        err = capsys.readouterr().err
        assert "Unknown tool" in err
        assert "alpha" in err and "beta" in err  # 候选清单随报

    def test_private_name_rejected(self, monkeypatch, capsys):
        _patch_eps(monkeypatch)  # 空注册表：表无名即不可达
        monkeypatch.setattr("sys.argv", ["mflowy", "_private"])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 1

    def test_runtime_error_exit_one(self, monkeypatch, capsys):
        def boom(a: Annotated[int, Field(description="甲")] = 1):
            raise RuntimeError("工具体内爆")

        _patch_eps(monkeypatch, boom=boom)
        monkeypatch.setattr("sys.argv", ["mflowy", "boom"])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 1
        assert "工具体内爆" in capsys.readouterr().err

    def test_query_result_echoed_stdout(self, monkeypatch, capsys):
        _patch_eps(monkeypatch, q=lambda: [{"step": "load"}])
        monkeypatch.setattr("sys.argv", ["mflowy", "q"])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        assert ei.value.code == 0
        assert '"step": "load"' in capsys.readouterr().out


class TestGroupedHelp:
    def test_bare_and_help_print_grouped_listing(self, monkeypatch, capsys):
        """mflowy 裸调 / -h → 分组工具清单 + 一行介（stdout，exit 0——裸调不视为错误）。"""
        _patch_eps(monkeypatch, probe_tool=_probe_tool)
        for argv in ([], ["-h"], ["--help"]):
            monkeypatch.setattr("sys.argv", ["mflowy", *argv])
            cli.main()  # 零参数/帮助 = 正常返回（exit 0）
            out = capsys.readouterr().out
            assert "probe_tool" in out
            assert "测试旗标编译" in out  # 一行介在场

    def test_grouping_by_owner_distribution(self, monkeypatch, capsys):
        """分组按 entry point 模块前缀（mflowy.app/data/model/bo）——单源可推导。"""
        plat = SimpleNamespace(name="plat", value="mflowy.app.tools:plat", load=lambda: _probe_tool)
        mdl = SimpleNamespace(name="mdl", value="mflowy.model.tools:mdl", load=lambda: _probe_tool)
        _patch_eps(monkeypatch, plat, mdl)
        monkeypatch.setattr("sys.argv", ["mflowy"])
        cli.main()
        out = capsys.readouterr().out
        assert "平台" in out and "建模域" in out

    def test_one_liner_folds_first_bullet_continuations(self):
        assert "测试旗标编译" in cli._one_liner(_probe_tool)

    def test_real_registry_help_lists_core_tools(self, capsys, monkeypatch):
        """真源全表：核心工具名全部入列（词表=运行期目录的同型断言）。"""
        monkeypatch.setattr("sys.argv", ["mflowy"])
        cli.main()
        out = capsys.readouterr().out
        for name in ("list_nodes", "modeling", "data_load_or_start", "bo_ask", "login"):
            assert name in out, f"分组清单缺 {name}"
