"""同意门端到端：内存传输 + 真实 ClientSession elicitation，验证首问/持久化/不再问/接线。"""

import asyncio
import json
import sys
import types as pytypes

import pytest
from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.server.mcpserver import MCPServer
from mcp.shared.memory import create_client_server_memory_streams
from mflowy.mcp import telemetry


@pytest.fixture(autouse=True)
def _isolate_telemetry_state(tmp_path, monkeypatch):
    """@cache 的 settings/api_key 路径与 wire_agentcat 的 synchronized_once 闭包均跨用例缓存——
    每用例重置并隔离 HOME。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    telemetry.settings_path.cache_clear()
    telemetry.api_key_path.cache_clear()
    telemetry._read_api_key.cache_clear()
    import mflowy.utils.wraps as wraps

    monkeypatch.setattr(telemetry, "wire_agentcat", wraps.synchronized_once()(telemetry.wire_agentcat.__wrapped__))
    yield
    telemetry.settings_path.cache_clear()
    telemetry.api_key_path.cache_clear()
    telemetry._read_api_key.cache_clear()


def _make_server() -> MCPServer:
    server = MCPServer(name="t")

    @server.tool()
    def ping() -> str:
        return "pong"

    return server


async def _call(server: MCPServer, elicitation_callback, *, calls=1, tool="ping", args=None, notifications=None):
    """内存传输上跑真实协议：initialize → N 次 tools/call。"""
    async with create_client_server_memory_streams() as (
        (client_reader, client_writer),
        (server_reader, server_writer),
    ):
        lowlevel = server._lowlevel_server
        server_task = asyncio.create_task(
            lowlevel.run(server_reader, server_writer, lowlevel.create_initialization_options())
        )
        try:
            async with ClientSession(
                client_reader,
                client_writer,
                elicitation_callback=elicitation_callback,
                message_handler=(
                    lambda msg: (
                        notifications.append(msg) if notifications is not None and hasattr(msg, "method") else None
                    )
                )
                if notifications is not None
                else None,
            ) as client:
                await client.initialize()
                return [await client.call_tool(tool, args or {}) for _ in range(calls)]
        finally:
            server_task.cancel()
            try:
                await server_task
            except (asyncio.CancelledError, Exception):
                pass


def _fake_agentcat():
    """sys.modules 注入的假 agentcat：记录 track() 调用；模块面补齐到 _patch_otlp_exporter 可达。"""
    mod = pytypes.ModuleType("agentcat")
    calls: list = []

    def track(server, project_id=None, options=None):
        calls.append((server, project_id, options))
        return server

    class AgentCatOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    types_mod = pytypes.ModuleType("agentcat.types")

    class OTLPExporterConfig(dict):
        pass

    class Event:
        pass

    types_mod.OTLPExporterConfig = OTLPExporterConfig
    types_mod.Event = Event

    modules_mod = pytypes.ModuleType("agentcat.modules")
    event_queue_mod = pytypes.ModuleType("agentcat.modules.event_queue")
    exporters_mod = pytypes.ModuleType("agentcat.modules.exporters")
    otlp_mod = pytypes.ModuleType("agentcat.modules.exporters.otlp")

    class OTLPExporter:
        pass

    otlp_mod.OTLPExporter = OTLPExporter
    exporters_mod.otlp = otlp_mod
    modules_mod.event_queue = event_queue_mod
    modules_mod.exporters = exporters_mod

    mod.track = track
    mod.AgentCatOptions = AgentCatOptions
    mod.types = types_mod
    mod.modules = modules_mod
    mod.calls = calls
    return mod


def _install_fake_agentcat(monkeypatch) -> list:
    fake = _fake_agentcat()
    monkeypatch.setitem(sys.modules, "agentcat", fake)
    monkeypatch.setitem(sys.modules, "agentcat.types", fake.types)
    monkeypatch.setitem(sys.modules, "agentcat.modules", fake.modules)
    monkeypatch.setitem(sys.modules, "agentcat.modules.event_queue", fake.modules.event_queue)
    monkeypatch.setitem(sys.modules, "agentcat.modules.exporters", fake.modules.exporters)
    monkeypatch.setitem(sys.modules, "agentcat.modules.exporters.otlp", fake.modules.exporters.otlp)
    return fake.calls


def test_gate_asks_once_persists_accept_and_wires_agentcat(tmp_path, monkeypatch):
    """接受 → 写 settings.json + 接线 agentcat（纯 OTLP）+ 广播 tools/list_changed + 两次调用只问一次。"""
    calls = _install_fake_agentcat(monkeypatch)
    monkeypatch.delenv("MFLOWY_TELEMETRY", raising=False)
    monkeypatch.setattr(telemetry, "DEFAULT_ENDPOINT", "http://otel:4318")

    server = _make_server()
    telemetry.install_consent_middleware(server)

    asked = []
    notifications = []

    async def on_elicit(context, params):
        asked.append(params)
        return mcp_types.ElicitResult(action="accept")

    results = asyncio.run(_call(server, on_elicit, calls=2, notifications=notifications))

    assert [r.content[0].text for r in results] == ["pong", "pong"]  # 工具照常执行
    assert len(asked) == 1  # 只问一次
    assert asked[0].message  # 询问带文案
    data = json.loads((tmp_path / ".mflowy" / "settings.json").read_text(encoding="utf-8"))
    assert data["telemetry"] is True

    assert len(calls) == 1  # 同意后接线
    methods = [n.method for n in notifications if hasattr(n, "method")]
    assert "notifications/tools/list_changed" in methods  # 接线后 schema 已注入，通知客户端重拉
    tracked_server, project_id, options = calls[0]
    assert tracked_server is server
    assert project_id is None  # 纯 OTLP 模式，不经 AgentCat 云
    assert options.exporters["otlp"]["endpoint"] == "http://otel:4318/traces"
    assert options.enable_report_missing is True
    assert options.enable_tool_call_context is True
    assert options.disable_diagnostics is True


def test_gate_decline_persists_off_and_never_asks_again(tmp_path, monkeypatch):
    """decline → 持久化 off、不接线；同进程后续调用与下次启动（新 server）都不再问。"""
    calls = _install_fake_agentcat(monkeypatch)
    monkeypatch.delenv("MFLOWY_TELEMETRY", raising=False)

    server = _make_server()
    telemetry.install_consent_middleware(server)

    asked = []

    async def decline(context, params):
        asked.append(params)
        return mcp_types.ElicitResult(action="decline")

    results = asyncio.run(_call(server, decline, calls=2))
    assert [r.content[0].text for r in results] == ["pong", "pong"]
    assert len(asked) == 1  # 第二次调用不再问（同进程）
    data = json.loads((tmp_path / ".mflowy" / "settings.json").read_text(encoding="utf-8"))
    assert data["telemetry"] is False
    assert calls == []  # 未接线

    # 下次启动 = 新 server + 新门：settings 已落盘 off → 直接不问
    server2 = _make_server()
    telemetry.install_consent_middleware(server2)

    async def would_ask(context, params):
        raise AssertionError("不应再次询问")

    asyncio.run(_call(server2, would_ask))
    assert calls == []


def test_gate_cancel_treated_as_accept(tmp_path, monkeypatch):
    """cancel（用户关闭弹窗未做选择）→ 按同意处理：落盘 on 并接线。"""
    calls = _install_fake_agentcat(monkeypatch)
    monkeypatch.delenv("MFLOWY_TELEMETRY", raising=False)

    server = _make_server()
    telemetry.install_consent_middleware(server)

    async def cancel(context, params):
        return mcp_types.ElicitResult(action="cancel")

    asyncio.run(_call(server, cancel))
    data = json.loads((tmp_path / ".mflowy" / "settings.json").read_text(encoding="utf-8"))
    assert data["telemetry"] is True
    assert len(calls) == 1  # 视为同意 → 接线


def test_gate_elicitation_failure_runs_off_without_persisting(tmp_path, monkeypatch, caplog):
    """客户端不支持 elicitation（提问抛错）→ 不落盘、不接线、工具不受影响。"""
    calls = _install_fake_agentcat(monkeypatch)
    monkeypatch.delenv("MFLOWY_TELEMETRY", raising=False)

    server = _make_server()
    telemetry.install_consent_middleware(server)

    async def broken(context, params):
        raise RuntimeError("no elicitation support")

    import logging

    with caplog.at_level(logging.DEBUG, logger="mflowy.mcp.telemetry"):
        result = asyncio.run(_call(server, broken))

    assert result[0].content[0].text == "pong"  # 工具照常
    assert not (tmp_path / ".mflowy" / "settings.json").exists()  # 未持久化（没能问到）
    assert calls == []
    assert any("elicitation" in r.message for r in caplog.records)


def test_main_on_wires_immediately(monkeypatch):
    """MFLOWY_TELEMETRY=on：main() 启动即接线 agentcat，不装同意门。"""
    from mflowy.mcp import server as server_module

    calls = {"register": [], "wire": [], "install": []}

    class FakeMCP:
        def __init__(self, **kw):
            calls["mcp_kwargs"] = kw

        def run(self):
            calls["ran"] = True

    monkeypatch.setattr(server_module, "MCPServer", FakeMCP)
    monkeypatch.setattr(server_module, "register", lambda mcp: calls["register"].append(mcp))
    monkeypatch.setattr(server_module.telemetry, "resolve_telemetry", lambda: "on")
    monkeypatch.setattr(server_module.telemetry, "wire_agentcat", lambda s: calls["wire"].append(s))
    monkeypatch.setattr(server_module.telemetry, "install_consent_middleware", lambda s: calls["install"].append(s))

    server_module.main()

    assert calls["ran"] is True
    assert calls["register"], "工具应注册"
    assert len(calls["wire"]) == 1
    assert calls["install"] == []


def test_main_ask_installs_consent_middleware(monkeypatch):
    """未决：main() 装同意门（不接线），同意后才接线（见 test_gate_asks_once...）。"""
    from mflowy.mcp import server as server_module

    calls = {"register": [], "wire": [], "install": []}

    class FakeMCP:
        def __init__(self, **kw):
            calls["mcp_kwargs"] = kw

        def run(self):
            calls["ran"] = True

    monkeypatch.setattr(server_module, "MCPServer", FakeMCP)
    monkeypatch.setattr(server_module, "register", lambda mcp: calls["register"].append(mcp))
    monkeypatch.setattr(server_module.telemetry, "resolve_telemetry", lambda: "ask")
    monkeypatch.setattr(server_module.telemetry, "wire_agentcat", lambda s: calls["wire"].append(s))
    monkeypatch.setattr(server_module.telemetry, "install_consent_middleware", lambda s: calls["install"].append(s))

    server_module.main()

    assert calls["ran"] is True
    assert len(calls["install"]) == 1  # 装门不接线
    assert calls["wire"] == []


def test_main_off_is_clean(monkeypatch):
    """off：既不装门也不接线，server 照常启动。"""
    from mflowy.mcp import server as server_module

    calls = {"register": [], "wire": [], "install": []}

    class FakeMCP:
        def __init__(self, **kw):
            calls["mcp_kwargs"] = kw

        def run(self):
            calls["ran"] = True

    monkeypatch.setattr(server_module, "MCPServer", FakeMCP)
    monkeypatch.setattr(server_module, "register", lambda mcp: calls["register"].append(mcp))
    monkeypatch.setattr(server_module.telemetry, "resolve_telemetry", lambda: "off")
    monkeypatch.setattr(server_module.telemetry, "wire_agentcat", lambda s: calls["wire"].append(s))
    monkeypatch.setattr(server_module.telemetry, "install_consent_middleware", lambda s: calls["install"].append(s))

    server_module.main()

    assert calls["ran"] is True
    assert calls["register"]
    assert calls["wire"] == [] and calls["install"] == []


def test_real_agentcat_otlp_includes_tool_io(tmp_path, monkeypatch):
    """真实 agentcat → 本地 OTLP sink：span 必须携带 mcp.parameters / mcp.response（全量 I/O 契约）。"""
    import importlib.util
    import threading
    import time as time_mod
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    if importlib.util.find_spec("agentcat") is None:
        pytest.skip("agentcat 未安装")

    recorded: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            recorded.append(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    sink = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=sink.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("MFLOWY_TELEMETRY", "on")
        monkeypatch.setattr(telemetry, "DEFAULT_ENDPOINT", f"http://127.0.0.1:{sink.server_address[1]}")

        server = MCPServer(name="t")

        @server.tool()
        def echo(x: str) -> str:
            return f"echo:{x}"

        telemetry.wire_agentcat(server)
        result = asyncio.run(_call(server, None, tool="echo", args={"x": "diag", "context": "回归测试意图自述"}))
        assert result[0].is_error is False

        deadline = time_mod.monotonic() + 25  # agentcat 批量窗口 5-10s，留裕量
        attrs = None
        while time_mod.monotonic() < deadline and attrs is None:
            for body in recorded:
                parsed = json.loads(body)
                if "resourceSpans" not in parsed:
                    continue  # 同桩注册请求（/device/register）体，非 OTLP
                for rs in parsed["resourceSpans"]:
                    for ss in rs["scopeSpans"]:
                        for sp in ss["spans"]:
                            found = {a["key"]: a["value"].get("stringValue", "") for a in sp.get("attributes", [])}
                            if found.get("mcp.resource_name") == "echo":
                                attrs = found
            if attrs is None:
                time_mod.sleep(0.5)
        assert attrs is not None, f"未捕获 echo span，posts={len(recorded)}"
        # context/session_id 为 agentcat 注入项：前者落 user_intent，后者有专属属性，均不入参
        assert json.loads(attrs["mcp.parameters"]) == {"x": "diag"}
        assert attrs["mcp.user_intent"] == "回归测试意图自述"
        assert "diag" in attrs["mcp.response"]
        assert attrs.get("mcp.result_type") == "complete"  # 新增诊断属性（真值才出现）
        assert "mcp.is_error" not in attrs  # 成功调用不携带错误标记
    finally:
        sink.shutdown()
