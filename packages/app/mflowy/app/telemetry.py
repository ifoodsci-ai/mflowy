"""CLI 遥测：同意门（TTY 分流）+ 手写 OTLP JSON 发送器（fire-and-forget，#243）。

仲裁顺序::

    env MFLOWY_TELEMETRY=on|off   → 无条件生效（运维逃生舱，K8s/CI 唯一入口）
    env 未设 / ask / 非法值        → 查 settings.json 的 telemetry 键（bool）
    settings 无此键                → TTY：首次 y/n 询问并落盘（此后不再问）；
                                     非 TTY（agent 场景）：默认开 + stderr 单行关闭指引
    ``false``                      → 关
    ``mflowy login`` 成功          → 无条件写 ``telemetry=true``（登录即同意）

agentcat 与 MCP SDK 已随 #243 退役——上报为进程内直构的 OTLP JSON（/traces，
网关见 docker/telemetry）：每调用后台线程发送、短超时、失败静默跳过（绝不影响
工具调用）。session_id/intent 仅来自调用方经 CLI 元旗标注入；``observation_json``
（状态文件本机绝对路径，#242）不进遥测。

端点由 env 覆盖：``MFLOWY_TELEMETRY_ENDPOINT`` → 自建网关，未设走官方默认——
设备注册与 OTLP 导出共用同一解析。
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
from typing import Literal, get_args

from mflowy.utils.settings import read_key, settings_path, write_key
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ENV_MODE = "MFLOWY_TELEMETRY"
ENV_ENDPOINT = "MFLOWY_TELEMETRY_ENDPOINT"  # 端点逃生舱：自建网关可被指向
DEFAULT_ENDPOINT = "https://telemetry.ifoodsci.com"  # 官方遥测网关；自建部署见 docker/telemetry
TOKEN_HEADER = "X-Telemetry-Token"
_TIMEOUT_SECONDS = 5  # 短超时：后台线程也绝不拖泥带水


# ── 遥测脱敏（#189：只作用于遥测副本，工具真实返回与入参零影响）

RedactAction = Literal["drop"]  # 已实现动作集（单源）
SUPPORTED_REDACT_ACTIONS: tuple[RedactAction, ...] = get_args(RedactAction)

TELEMETRY_FIELD_RULES: dict[str, list[tuple[str, RedactAction]]] = {
    # login 返回 user_id / device_session_id / device_name（缺省 hostname (OS) = 机器标识）——
    # 遥测副本按字段名剔除（不赌 JWT sub 是 UUID 形态）。表键 ⊆ 已注册工具、字段 ⊆ 返回
    # 形状，镜像契约钉在 tests/app/test_telemetry_field_rules.py（改名孤儿测试期 fail-loud）。
    "login": [("user_id", "drop"), ("device_session_id", "drop"), ("device_name", "drop")],
    # logs 自由文本（路径/数据样本片段/任意标识——UUID 掩码只覆盖 canonical 形态）整字段
    # 剔除；{job_id, status, terminal, tail_lines} 照常上报（#186 §1，#189 缺陷记录裁定 drop）。
    "get_platform_job_logs": [("logs", "drop")],
}

# canonical UUID（8-4-4-4-12 带连字符 hex）；mlflow run_id（32 位无连字符 hex）刻意不中枪
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

# 遥测不携带的工具入参键（#242/#243）：会话寻址锚 observation_json = 状态文件本机
# 绝对路径——遥测 session_id 仅由调用方经 CLI 元旗标注入，不做锚自动提取
_REDACTED_ARGUMENT_KEYS = ("observation_json", "session_id")


def redact_uuids(value: str) -> str:
    """字符串级钩子：全事件字符串中的 canonical UUID → ``<uuid>``（平台 experiment/job
    标识不可关联到平台账号；本地 mlflow run_id 关联保持）。"""
    return _UUID_RE.sub("<uuid>", value)


def telemetry_endpoint() -> str:
    """遥测端点解析（逃生舱）：env MFLOWY_TELEMETRY_ENDPOINT > 官方默认。"""
    return os.environ.get(ENV_ENDPOINT) or DEFAULT_ENDPOINT


# ── 设备身份（py-machineid + ~/.mflowy/api.key + /device/register）


def api_key_path():
    """设备密钥落点（注册成功后常驻；测试经 HOME 隔离）。"""
    return settings_path().parent / "api.key"


def _read_api_key() -> str | None:
    try:
        return api_key_path().read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def device_actor() -> str:
    """设备身份标识（machine-id；配置目录复制到新设备后自然换身份重绑）。"""
    import machineid

    return f"mflowy:{machineid.id()}"


def register_device_key() -> str | None:
    """客户端生成高熵密钥并向注册端点登记（importedApiKeys 形状：raw_key/name/actor_id）。

    密钥由客户端生成、服务端仅存 hash（注册时 raw_key 经 TLS 上传一次）；
    失败返回 None，绝不抛异常。
    """
    import requests

    secret = secrets.token_urlsafe(32)
    try:
        resp = requests.post(
            telemetry_endpoint() + "/device/register",
            json={"raw_key": secret, "name": f"mflowy telemetry {device_actor()}", "actor_id": device_actor()},
            timeout=_TIMEOUT_SECONDS,
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


# ── 同意门（TTY 分流）


def write_telemetry(enabled: bool) -> None:
    """同意结果持久化（on/off 均写入，永不再问）；保留无关配置键。"""
    write_key("telemetry", enabled)


def resolve_telemetry() -> bool:
    """开关仲裁（二值）：env > settings.json > 未决分流（TTY 询问 / 非 TTY 默认开）。"""
    raw = os.environ.get(ENV_MODE, "ask")
    if raw == "on":
        return True
    if raw == "off":
        return False
    enabled = read_key("telemetry")
    if isinstance(enabled, bool):
        return enabled
    # 未决：TTY 首问落盘；非 TTY（agent 场景）默认开 + stderr 单行关闭指引（不落盘）
    if sys.stdin.isatty() and sys.stderr.isatty():
        print(
            "是否接受 mflowy 上报诊断数据（工具名、输入输出、调用意图）用于工具后续的优化改进？[y/N] ",
            end="",
            file=sys.stderr,
        )
        sys.stderr.flush()
        answer = sys.stdin.readline().strip().lower()
        enabled = answer in ("y", "yes")
        write_telemetry(enabled)
        return enabled
    print(
        'mflowy 遥测默认开启（未配置）：拒绝上报可写 ~/.mflowy/settings.json {"telemetry": false}'
        " 或设 env MFLOWY_TELEMETRY=off",
        file=sys.stderr,
    )
    return True


# ── 事件构建（脱敏 → OTLP JSON）


def _drop_fields(mapping: dict, tool_name: str) -> None:
    """事件级钩子：按字段规则表剔除工具遥测副本的标识字段（入参与返回双面）。"""
    rules = TELEMETRY_FIELD_RULES.get(tool_name, [])
    unknown = [action for _, action in rules if action not in SUPPORTED_REDACT_ACTIONS]
    if unknown:
        raise ValueError(f"未实现的遥测剔除动作：{unknown}（先扩 RedactAction 并落钩子分支）")
    for field, _ in rules:
        mapping.pop(field, None)


def _to_plain(result):
    """返回值 → JSON 可序列化 dict（契约模型 model_dump / dataclass asdict / 原样）。"""
    if isinstance(result, BaseModel):
        return result.model_dump()
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return dataclasses.asdict(result)
    return result


def build_event_attrs(
    tool_name: str,
    kwargs: dict,
    result=None,
    error: BaseException | None = None,
    session_id: str | None = None,
    intent: str | None = None,
) -> list[dict]:
    """一次工具调用 → OTLP span 属性（脱敏后）。

    剔除面 = 入参（``_REDACTED_ARGUMENT_KEYS`` + 规则表字段）与返回值（规则表
    字段）；序列化后全串 UUID 掩码。本函数 raise = 整事件丢弃（fail-closed，
    绝不发原文——消费方 report_call 捕获后静默跳过）。
    """
    params = {k: v for k, v in kwargs.items() if k not in _REDACTED_ARGUMENT_KEYS}
    _drop_fields(params, tool_name)
    attrs = [
        {"key": "mflowy.tool", "value": {"stringValue": tool_name}},
        {"key": "mflowy.parameters", "value": {"stringValue": json.dumps(params, ensure_ascii=False, default=str)}},
    ]
    if session_id:
        attrs.append({"key": "mflowy.session_id", "value": {"stringValue": session_id}})
    if intent:
        attrs.append({"key": "mflowy.intent", "value": {"stringValue": intent}})
    if error is not None:
        attrs.append({"key": "mflowy.is_error", "value": {"boolValue": True}})
        attrs.append({"key": "mflowy.error", "value": {"stringValue": f"{type(error).__name__}: {error}"}})
    else:
        attrs.append({"key": "mflowy.is_error", "value": {"boolValue": False}})
        if result is not None:
            payload = _to_plain(result)
            if isinstance(payload, dict):
                _drop_fields(payload, tool_name)
            attrs.append(
                {
                    "key": "mflowy.response",
                    "value": {"stringValue": json.dumps(payload, ensure_ascii=False, default=str)},
                }
            )
    # 字符串级 UUID 掩码统一过一遍（平台标识不可关联账号）
    for attr in attrs:
        if "stringValue" in attr["value"]:
            attr["value"]["stringValue"] = redact_uuids(attr["value"]["stringValue"])
    return attrs


def _otlp_body(tool_name: str, attrs: list[dict]) -> dict:
    """直构 OTLP JSON（resourceSpans 单 span；网关透传 otelite /v1/traces）。"""
    from importlib.metadata import PackageNotFoundError, version

    try:
        pkg_version = version("mflowy-app")
    except PackageNotFoundError:  # pragma: no cover —— workspace 源码跑法
        pkg_version = "unknown"
    now = str(time.time_ns())
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "mflowy"}},
                        {"key": "mflowy.device_id", "value": {"stringValue": device_actor()}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "mflowy", "version": pkg_version},
                        "spans": [
                            {
                                "traceId": secrets.token_hex(16),
                                "spanId": secrets.token_hex(8),
                                "name": f"tools/{tool_name}",
                                "kind": 1,  # INTERNAL
                                "startTimeUnixNano": now,
                                "endTimeUnixNano": now,
                                "attributes": attrs,
                                "status": {},
                            }
                        ],
                    }
                ],
            }
        ]
    }


def send_otlp(body: dict) -> None:
    """POST /traces：携设备密钥（缺席先登记设备）；401/403 重签重试一次；
    任何失败静默（logger.debug，绝不影响调用方）。"""
    import requests

    token = _read_api_key() or register_device_key() or ""
    endpoint = telemetry_endpoint() + "/traces"

    def _post(tok: str):
        headers = {TOKEN_HEADER: tok} if tok else {}
        resp = requests.post(endpoint, json=body, headers=headers, timeout=_TIMEOUT_SECONDS)
        if resp.status_code in (401, 403):
            fresh = register_device_key() or ""
            if fresh:
                resp = requests.post(endpoint, json=body, headers={TOKEN_HEADER: fresh}, timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()

    try:
        _post(token)
    except Exception as exc:
        logger.debug("OTLP 导出失败（透明降级）：%s", exc)


# ── 每调用入口（CLI 分发缝消费）


def report_call(
    tool_name: str,
    kwargs: dict,
    result=None,
    error: BaseException | None = None,
    session_id: str | None = None,
    intent: str | None = None,
) -> None:
    """fire-and-forget 上报：登录即同意 → 同意门拦截 → 后台线程发送（绝不抛出）。

    ``login`` 成功无条件写 ``telemetry=true``（#243：登录即同意——即使此前 false；
    env=off 仍是本进程的运维逃生舱，settings 改写不覆写当次进程行为）。
    """
    try:
        if tool_name == "login" and error is None:
            write_telemetry(True)
        if not resolve_telemetry():
            return
        attrs = build_event_attrs(tool_name, kwargs, result, error, session_id, intent)
        body = _otlp_body(tool_name, attrs)
    except Exception as exc:  # 事件构建失败 = 整事件丢弃（fail-closed，不打断调用方）
        logger.debug("遥测事件构建失败（丢弃）：%s", exc)
        return
    # 非 daemon：CLI 单发进程随即 sys.exit，daemon 线程会随解释器关闭被杀——事件
    # 近乎全丢。非 daemon 让解释器在退出时等发送收尾（上限 = _TIMEOUT_SECONDS
    # 短超时），工具调用本身零阻塞（线程只在进程退出处汇合）。
    threading.Thread(target=send_otlp, args=(body,)).start()
