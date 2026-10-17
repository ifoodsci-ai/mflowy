"""设备密钥绑定（方案A：客户端生成对称秘密 + importedApiKeys 幂等注册）。

真实 agentcat + docker/telemetry 单服务桩（同一端口同时服务 /device/register 与
/traces，端点即固定常量 DEFAULT_ENDPOINT，测试经 monkeypatch 指向桩）：秘密由
客户端生成，注册请求携带 raw_key，服务端不回传 secret；api.key 存的是客户端自生成的值。
"""

import asyncio
import json
import threading
import time as time_mod
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
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


class _Stub:
    """可编程桩服务：记录请求，按脚本返回。"""

    def __init__(self, handler_cls):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _gateway_stub(*, register_fail=False, reject_first=False):
    """docker/telemetry 单服务桩：同端口提供 /device/register 与 /traces。"""
    register_records = []
    sink_records = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
            token = self.headers.get("X-Telemetry-Token")
            if self.path == "/device/register":
                status, payload = (
                    (500, {"error": "register down"})
                    if register_fail
                    else (200, {"key_id": f"key_{len(register_records)}"})
                )
                register_records.append({"path": self.path, "body": body, "token": token})
            elif self.path == "/traces":
                status, payload = (401, {"error": "unauthorized"}) if reject_first and not sink_records else (200, {})
                sink_records.append({"path": self.path, "body": body, "token": token})
            else:
                status, payload = 404, {"error": "not found"}
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    stub = _Stub(Handler)
    stub.register_records = register_records
    stub.sink_records = sink_records
    return stub


async def _call_once():
    server = MCPServer(name="t")

    @server.tool()
    def ping() -> str:
        return "pong"

    telemetry.wire_agentcat(server)
    async with create_client_server_memory_streams() as (
        (client_reader, client_writer),
        (server_reader, server_writer),
    ):
        lowlevel = server._lowlevel_server
        server_task = asyncio.create_task(
            lowlevel.run(server_reader, server_writer, lowlevel.create_initialization_options())
        )
        try:
            async with ClientSession(client_reader, client_writer) as client:
                await client.initialize()
                r = await client.call_tool("ping", {})
                assert r.is_error is False
        finally:
            server_task.cancel()
            try:
                await server_task
            except BaseException:
                pass


def _run_tool(monkeypatch, tmp_path, stub):
    """端点固定为常量（stdio MCP 契约）：monkeypatch DEFAULT_ENDPOINT 指向桩服务裸 base。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MFLOWY_TELEMETRY", "on")
    monkeypatch.setattr(telemetry, "DEFAULT_ENDPOINT", f"http://127.0.0.1:{stub.port}")
    asyncio.run(_call_once())
    deadline = time_mod.monotonic() + 25
    while time_mod.monotonic() < deadline and not stub.sink_records:
        time_mod.sleep(0.5)


# ── 设备密钥绑定（真 agentcat + 单服务桩）──────────────────────────────────


def test_missing_key_generates_and_registers_locally_owned_secret(tmp_path, monkeypatch):
    """无 api.key → 客户端生成高熵秘密 → 注册携带 raw_key（响应不回传）→ 落盘即自生成值。"""
    stub = _gateway_stub()
    try:
        _run_tool(monkeypatch, tmp_path, stub)
        assert len(stub.register_records) == 1, f"注册请求数 {len(stub.register_records)} != 1"
        req = json.loads(stub.register_records[0]["body"])
        assert len(req["raw_key"]) >= 32, "raw_key 应为高熵秘密"
        assert req["name"] and req["actor_id"]  # actor_id 含设备指纹
        key_file = tmp_path / ".mflowy" / "api.key"
        saved = key_file.read_text(encoding="utf-8").strip()
        assert saved == req["raw_key"], "api.key 应存客户端自生成秘密（非服务端下发）"
        assert stub.sink_records, "OTLP span 未到达 sink"
        assert stub.sink_records[0]["token"] == saved
    finally:
        stub.close()


def test_existing_key_no_register_and_header_attached(tmp_path, monkeypatch):
    """已有 api.key → 不注册，直接带既有秘密发送。"""
    key_file = tmp_path / ".mflowy" / "api.key"
    key_file.parent.mkdir(parents=True)
    key_file.write_text("sk_existing_local\n", encoding="utf-8")
    stub = _gateway_stub()
    try:
        _run_tool(monkeypatch, tmp_path, stub)
        assert stub.register_records == [], "不应发生注册"
        assert stub.sink_records[0]["token"] == "sk_existing_local"
    finally:
        stub.close()


def test_401_triggers_reregister_with_fresh_secret_and_retry(tmp_path, monkeypatch):
    """首发 401 → 重新生成秘密（≠旧值）→ 再注册 → 单次重试送达。"""
    key_file = tmp_path / ".mflowy" / "api.key"
    key_file.parent.mkdir(parents=True)
    key_file.write_text("sk_stale\n", encoding="utf-8")
    stub = _gateway_stub(reject_first=True)
    try:
        _run_tool(monkeypatch, tmp_path, stub)
        assert len(stub.register_records) == 1, "401 后应恰好再注册一次"
        fresh = json.loads(stub.register_records[0]["body"])["raw_key"]
        assert fresh != "sk_stale", "重注册应生成全新秘密"
        assert key_file.read_text(encoding="utf-8").strip() == fresh
        tokens = [r["token"] for r in stub.sink_records]
        assert tokens[0] == "sk_stale" and tokens[-1] == fresh
        assert len(stub.sink_records) == 2, f"期望首发+重试共 2 次，实际 {len(stub.sink_records)}"
    finally:
        stub.close()


def test_register_down_single_attempt_and_never_breaks_tools(tmp_path, monkeypatch):
    """注册端点不可达 → 工具照常、仅尝试一次（进程级退避）、不落盘。"""
    stub = _gateway_stub(register_fail=True)
    try:
        _run_tool(monkeypatch, tmp_path, stub)
        time_mod.sleep(3)  # 留出可能的重试窗口，验证不轰端点
        assert len(stub.register_records) == 1, f"注册尝试 {len(stub.register_records)} 次，应进程内只试一次"
        assert not (tmp_path / ".mflowy" / "api.key").exists()
    finally:
        stub.close()
