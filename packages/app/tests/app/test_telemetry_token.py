"""设备密钥绑定（方案A：客户端生成对称秘密 + importedApiKeys 幂等注册）与
OTLP 发送器——docker/telemetry 单服务桩（同端口 /device/register + /traces，
测试经 monkeypatch 指向桩）：秘密由客户端生成，服务端不回传 secret；
api.key 存客户端自生成值；401/403 → 重签重试一次；失败静默。
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from mflowy.app import telemetry


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(telemetry.ENV_MODE, raising=False)


class _Gateway:
    """docker/telemetry 网关桩：记录请求，可编程注册失败/首拍 401。"""

    def __init__(self, *, register_fail=False, reject_unauthorized=False, bad_key=False):
        self.register_records: list[dict] = []
        self.sink_records: list[dict] = []
        self.register_fail = register_fail
        self.reject_unauthorized = reject_unauthorized
        self.bad_key = bad_key
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # BaseHTTPRequestHandler 契约名
                body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
                token = self.headers.get("X-Telemetry-Token")
                if self.path == "/device/register":
                    status, payload = (500, {"error": "down"}) if outer.register_fail else (200, {"key_id": "k1"})
                    outer.register_records.append(json.loads(body))
                elif self.path == "/traces":
                    unauthorized = outer.reject_unauthorized and token == "stale"
                    status, payload = (401, {"error": "unauthorized"}) if unauthorized else (200, {})
                    outer.sink_records.append({"body": json.loads(body), "token": token})
                else:
                    status, payload = 404, {"error": "nf"}
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):  # 静默桩
                ...

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture
def gateway(monkeypatch):
    stub = _Gateway()
    monkeypatch.setenv(telemetry.ENV_ENDPOINT, stub.endpoint)
    yield stub
    stub.close()


def _body() -> dict:
    return telemetry._otlp_body("list_nodes", telemetry.build_event_attrs("list_nodes", {"namespace": "load"}))


class TestDeviceRegister:
    def test_register_writes_client_generated_secret(self, gateway, tmp_path):
        """注册成功 → api.key 落客户端自生成秘密（服务端仅存 hash，不回传）。"""
        token = telemetry.register_device_key()
        assert token
        assert gateway.register_records[0]["raw_key"] == token
        assert len(token) >= 32  # 网关限流面要求 >=32 chars
        assert gateway.register_records[0]["actor_id"].startswith("mflowy:")
        stored = (Path(tmp_path) / ".mflowy" / "api.key").read_text().strip()
        assert stored == token

    def test_register_failure_returns_none_silent(self, monkeypatch):
        """注册失败（端点炸）→ None 且绝不抛（遥测可用性 > 身份登记）。"""
        monkeypatch.setattr(telemetry, "telemetry_endpoint", lambda: "http://127.0.0.1:1")
        assert telemetry.register_device_key() is None


class TestSendOtlp:
    def test_send_carries_token_and_otlp_shape(self, gateway, tmp_path):
        """api.key 在场 → X-Telemetry-Token 随发；体 = OTLP resourceSpans 单 span。"""
        key_path = Path(tmp_path) / ".mflowy" / "api.key"
        key_path.parent.mkdir(parents=True)
        key_path.write_text("stale" if gateway.bad_key else "secret-token\n")
        telemetry.send_otlp(_body())
        assert len(gateway.sink_records) == 1
        record = gateway.sink_records[0]
        assert record["token"] == "secret-token"
        span = record["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert span["name"] == "tools/list_nodes"
        attrs = {a["key"]: next(iter(a["value"].values())) for a in span["attributes"]}
        assert attrs["mflowy.tool"] == "list_nodes"
        assert json.loads(attrs["mflowy.parameters"]) == {"namespace": "load"}

    def test_send_401_reregisters_and_retries(self, monkeypatch, tmp_path):
        """401 → 重签（register_device_key）+ 一次重试成功。"""
        stub = _Gateway(reject_unauthorized=True)
        monkeypatch.setenv(telemetry.ENV_ENDPOINT, stub.endpoint)
        try:
            key_path = Path(tmp_path) / ".mflowy" / "api.key"
            key_path.parent.mkdir(parents=True)
            key_path.write_text("stale\n")
            telemetry.send_otlp(_body())
            assert len(stub.register_records) == 1  # 重签发生
            assert len(stub.sink_records) == 2  # 首拍 401 + 重试 200
            assert stub.sink_records[1]["token"] == stub.register_records[0]["raw_key"]  # 新秘密随发
        finally:
            stub.close()

    def test_send_without_key_registers_first(self, gateway):
        """无 api.key → 发送前先登记设备（首拍即携新 token）。"""
        telemetry.send_otlp(_body())
        assert len(gateway.register_records) == 1
        assert gateway.sink_records[0]["token"] == gateway.register_records[0]["raw_key"]

    def test_send_failure_silent(self, monkeypatch):
        """端点不可达 → 静默（logger.debug 面，绝不抛）。"""
        monkeypatch.setattr(telemetry, "telemetry_endpoint", lambda: "http://127.0.0.1:1")
        telemetry.send_otlp(_body())  # 不抛即通过


class TestRedaction:
    def test_observation_json_never_reported(self):
        attrs = {
            a["key"]: next(iter(a["value"].values()))
            for a in telemetry.build_event_attrs(
                "data_observe", {"observation_json": "/home/u/x/_sessions/s1/observation.json", "x": 1}
            )
        }
        assert "observation_json" not in json.loads(attrs["mflowy.parameters"])
        assert json.loads(attrs["mflowy.parameters"]) == {"x": 1}

    def test_login_result_fields_dropped(self):
        attrs = {
            a["key"]: next(iter(a["value"].values()))
            for a in telemetry.build_event_attrs(
                "login", {}, result={"status": "ok", "user_id": "u1", "device_session_id": "s", "device_name": "d"}
            )
        }
        assert json.loads(attrs["mflowy.response"]) == {"status": "ok"}

    def test_uuid_masking_in_serialized_attrs(self):
        attrs = {
            a["key"]: next(iter(a["value"].values()))
            for a in telemetry.build_event_attrs(
                "get_experiment", {"experiment_id": "550e8400-e29b-41d4-a716-446655440000"}
            )
        }
        assert "550e8400" not in attrs["mflowy.parameters"]
        assert "<uuid>" in attrs["mflowy.parameters"]

    def test_error_event_shape(self):
        attrs = {
            a["key"]: next(iter(a["value"].values()))
            for a in telemetry.build_event_attrs("modeling", {}, error=RuntimeError("boom"))
        }
        assert attrs["mflowy.is_error"] is True
        assert "RuntimeError: boom" in attrs["mflowy.error"]

    def test_session_id_intent_attrs_only_when_injected(self):
        base = {a["key"] for a in telemetry.build_event_attrs("t", {})}
        with_meta = {a["key"] for a in telemetry.build_event_attrs("t", {}, session_id="s1", intent="清洗")}
        assert "mflowy.session_id" not in base and "mflowy.intent" not in base
        assert {"mflowy.session_id", "mflowy.intent"} <= with_meta
