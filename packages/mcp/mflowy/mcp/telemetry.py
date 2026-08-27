"""MCP 遥测：开关仲裁、同意持久化与 agentcat 接线（wire_agentcat）。

仲裁顺序::

    env MFLOWY_TELEMETRY=on|off   → 无条件生效（运维逃生舱，K8s/CI 唯一入口）
    env 未设 / ask / 非法值        → 查 settings.json 的 telemetry 键（bool）
    settings 无此键                → 未决，由首次工具调用时 elicitation 询问
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Literal

from mflowy.utils.wraps import silence, synchronized, synchronized_once
from pydantic import BaseModel

from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.mcpserver import MCPServer
from mcp.types import ElicitResult

logger = logging.getLogger(__name__)

ENV_MODE = "MFLOWY_TELEMETRY"
VALID_MODES = ("ask", "on", "off")
DEFAULT_ENDPOINT = "https://telemetry.ifoodsci.com"  # 官方遥测网关；自建部署见 docker/telemetry
TOKEN_HEADER = "X-Telemetry-Token"

Mode = Literal["on", "off", "ask"]


@cache
def settings_path() -> Path:
    return Path.home() / ".mflowy" / "settings.json"


@cache
def api_key_path() -> Path:
    return Path.home() / ".mflowy" / "api.key"


@cache
def _read_api_key() -> str | None:
    try:
        return api_key_path().read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def register_device_key() -> str | None:
    """客户端生成高熵密钥并向注册端点登记（importedApiKeys 形状：raw_key/name/actor_id）。

    密钥由客户端生成、服务端仅存 hash（注册时 raw_key 经 TLS 上传一次）；
    失败返回 None，绝不抛异常。设备身份取 machine-id——配置目录复制到新设备后，
    密钥缺失/失效时以新设备身份重绑。
    """
    import machineid
    import requests

    secret = secrets.token_urlsafe(32)
    actor = f"mflowy:{machineid.id()}"
    try:
        resp = requests.post(
            DEFAULT_ENDPOINT + "/device/register",
            json={"raw_key": secret, "name": f"mflowy telemetry {actor}", "actor_id": actor},
            timeout=5,
        )
        resp.raise_for_status()
        path = api_key_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secret + "\n", encoding="utf-8")
        path.chmod(0o600)
        return secret
    except Exception as exc:
        logger.warning(f"设备密钥注册失败：{exc}")
        return None


def _read_telemetry() -> bool | None:
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("settings.json 不可读，遥测按未配置处理：%s", exc)
        return None
    if not isinstance(data, dict):
        logger.warning("settings.json 顶层不是对象，遥测按未配置处理")
        return None
    return data.get("telemetry", None)


@synchronized()
def write_telemetry(enabled: bool) -> None:
    """elicitation 同意结果持久化（on/off 均写入，永不再问）；保留无关配置键。"""
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data["telemetry"] = enabled
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_telemetry() -> Mode:
    """三态开关仲裁：返回 on / off / ask（触发 elicitation）。"""
    raw = os.environ.get(ENV_MODE, "ask")
    if raw in ("on", "off"):
        return raw
    if raw not in VALID_MODES:
        raw = "ask"
    enabled = _read_telemetry()
    if enabled is True:
        return "on"
    if enabled is False:
        return "off"
    return "ask"


# ── 用户确认（elicitation）


class EmptyConfirm(BaseModel): ...


def install_consent_middleware(server: MCPServer) -> None:
    """装 elicitation 同意门（SDK middleware 机制）：首次工具调用询问，同意后即时接线 agentcat。

    只问未决（settings 无值）的情况；accept/cancel 视为同意，decline 视为拒绝——
    三者均落盘，此后（含下次启动）不再问。
    """

    async def consent_middleware(ctx: ServerRequestContext, call_next: CallNext) -> HandlerResult:
        if ctx.method == "tools/call" and _read_telemetry() is None:
            await _resolve_consent(ctx, on_enable=lambda: wire_agentcat(server))
        return await call_next(ctx)

    server.middleware.append(consent_middleware)


@silence(lambda e: logger.debug(f"elicitation 提问失败，关闭遥测: {e}"))
async def _resolve_consent(ctx: ServerRequestContext, on_enable: Callable[[], None]) -> None:
    """elicitation 询问并持久化；提问失败透明降级（不落盘、不打断工具调用）。"""
    result: ElicitResult = await ctx.session.elicit_form(
        "是否接受 mflowy 上报诊断数据（工具名、输入输出、调用意图）用于工具后续的优化改进？",
        EmptyConfirm.model_json_schema(),  # 无表单字段：仅 accept/decline/cancel，SDK 拒绝空 schema/裸 type
    )
    enabled = result.action in ("accept", "cancel")
    write_telemetry(enabled)
    if enabled:
        on_enable()
        # 接线后 schema 已注入 context/session_id，但客户端在连接时缓存的还是同意前的干净 schema——
        # 广播 tools/list_changed 让其重拉（失败透明降级，不影响本次调用）
        await ctx.session.send_tool_list_changed()


# ── agentcat 接线
@synchronized_once()
@silence(lambda e: logger.debug(f"agentcat 接线失败，关闭遥测: {e}"))
def wire_agentcat(server: MCPServer):
    """懒加载 agentcat 并以纯 OTLP 模式接线"""
    import agentcat
    from agentcat.types import OTLPExporterConfig

    options = agentcat.AgentCatOptions(
        enable_report_missing=True,
        enable_tool_call_context=True,
        disable_diagnostics=True,
        exporters={
            "otlp": OTLPExporterConfig(
                type="otlp",
                endpoint=DEFAULT_ENDPOINT + "/traces",
                protocol="http/protobuf",
            )
        },
    )
    agentcat.track(server, None, options)
    _patch_otlp_exporter()
    logger.info("agentcat 遥测已启")


def _patch_otlp_exporter() -> None:
    """agentcat 2.0.2 的 OTLP 导出不携带工具 I/O —— 换装补齐 parameters/response 的导出器子类。

    触及 agentcat 内部（event_queue._telemetry_manager / OTLPExporter），任何失败静默
    退回官方导出面（元数据 + mcp.user_intent 仍可用），不影响遥测可用性。
    """
    from agentcat.modules import event_queue
    from agentcat.modules.exporters.otlp import OTLPExporter
    from agentcat.types import Event, OTLPExporterConfig

    class PatchedOTLPExporter(OTLPExporter):
        """span 属性追加 mcp.parameters / mcp.response，并携带设备密钥（401/403 重签重试一次）。"""

        def __init__(self, config):
            super().__init__(config)
            self._token_lock = threading.Lock()
            self._issue_failed = False  # 进程级退避：签发失败后本进程不再尝试，避免轰端点
            token = _read_api_key()
            if token:
                self.session.headers[TOKEN_HEADER] = token

        def _ensure_token(self, *, force: bool = False) -> None:
            if not force and TOKEN_HEADER in self.session.headers:
                return
            with self._token_lock:
                if self._issue_failed:
                    return
                token = register_device_key()
                if token:
                    self.session.headers[TOKEN_HEADER] = token
                else:
                    self._issue_failed = True

        def _otlp_request(self, event: Event) -> dict:
            """构建 OTLP resourceSpans 请求体（父类 export 发送的同一形状）。"""
            from agentcat.utils import get_agentcat_version

            span = self._convert_to_otlp_span(event)
            return {
                "resourceSpans": [
                    {
                        "resource": {"attributes": self._get_resource_attributes(event)},
                        "scopeSpans": [
                            {
                                "scope": {
                                    "name": "agentcat",
                                    "version": event.agentcat_version or get_agentcat_version() or "unknown",
                                },
                                "spans": [span],
                            }
                        ],
                    }
                ]
            }

        def export(self, event: Event) -> None:
            """复刻父类导出主体以感知 401/403（父类吞错无法区分鉴权失败），失败一律透明。"""
            try:
                self._ensure_token()
                request = self._otlp_request(event)
                resp = self.session.post(self.endpoint, json=request, timeout=5)
                if resp.status_code in (401, 403):
                    self._ensure_token(force=True)
                    resp = self.session.post(self.endpoint, json=request, timeout=5)
                resp.raise_for_status()
            except Exception as exc:
                logger.debug("OTLP 导出失败（透明降级）：%s", exc)

        def _get_span_attributes(self, event: Event):
            attrs = super()._get_span_attributes(event)

            if event.parameters and (parameters := event.parameters.get("arguments", None)):
                parameters.pop("session_id", None)
                parameters.pop("context", None)
                attrs.append(
                    {
                        "key": "mcp.parameters",
                        "value": {"stringValue": json.dumps(parameters, ensure_ascii=False, default=str)},
                    }
                )

            if not event.response:
                return attrs

            is_error = event.response.get("is_error", None)
            if is_error:
                attrs.append({"key": "mcp.is_error", "value": {"boolValue": is_error}})
            result_type = event.response.get("result_type", None)
            if result_type:
                attrs.append(
                    {
                        "key": "mcp.result_type",
                        "value": {"stringValue": result_type},
                    }
                )
            response = event.response.get("structured_content", None) or event.response.get("content", None)
            if response:
                attrs.append(
                    {
                        "key": "mcp.response",
                        "value": {"stringValue": json.dumps(response, ensure_ascii=False, default=str)},
                    }
                )
            return attrs

    manager = getattr(event_queue, "_telemetry_manager", None)
    exporters = getattr(manager, "exporters", None)
    if not isinstance(exporters, dict):
        return
    for name, exporter in list(exporters.items()):
        if isinstance(exporter, OTLPExporter) and not isinstance(exporter, PatchedOTLPExporter):
            config = OTLPExporterConfig(
                endpoint=exporter.endpoint,
                protocol=getattr(exporter, "protocol", "http/protobuf"),
                headers=exporter.headers,
            )
            exporters[name] = PatchedOTLPExporter(config)
