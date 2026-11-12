"""遥测字段规则表镜像契约：表键 ⊆ 已注册工具、字段 ⊆ 工具可见面、动作 ⊆ 已实现集。

剔除行是隐私契约（PRIVACY.md「不采集」按构造保证的组成件）——指向已改名/删除的
工具或字段即静默失效的脱敏，双向差集在此 fail-loud（红线 1/7）。
"""

import copy
import inspect
import json
from importlib.metadata import entry_points
from typing import Any

from mflowy.mcp import telemetry


def _registered_tools() -> set[str]:
    """已注册工具全集 = mflowy.tools entry points（server.register 同一真源）。"""
    return {ep.name for ep in entry_points(group="mflowy.tools")}


def test_rule_table_keys_are_registered_tools():
    orphans = set(telemetry.TELEMETRY_FIELD_RULES) - _registered_tools()
    assert not orphans, f"字段规则表指向未注册工具（改名孤儿？）: {sorted(orphans)}"


def test_rule_table_actions_implemented():
    """动作面以 telemetry 已实现集单源——表里出现未实现动作 = 剔除静默失效。"""
    for tool, rules in telemetry.TELEMETRY_FIELD_RULES.items():
        for field, action in rules:
            assert action in telemetry.SUPPORTED_REDACT_ACTIONS, (
                f"{tool}.{field} 动作 {action!r} 未实现（先扩已实现集并落钩子分支）"
            )


def test_login_rule_fields_in_visible_surfaces():
    """剔除字段必须咬合 login 可见面至少一面，否则改名孤儿静默失效。

    返回面单源 = Credentials.summary()；入参面单源 = 注册函数签名——钩子对两面
    同名字段做剔除，字段两面全 miss 即孤儿。
    """
    from mflowy.utils import auth

    summary = auth.Credentials(
        base_url="https://x",
        client_id="c",
        access_token="a",
        refresh_token="r",
        scope="s",
        user_id="u",
        session_id="sid",
        device_name="d",
        obtained_at=0.0,
        expires_at=0.0,
    ).summary("ok")
    login_fn = next(ep.load() for ep in entry_points(group="mflowy.tools") if ep.name == "login")
    params = set(inspect.signature(login_fn).parameters)
    for field, _ in telemetry.TELEMETRY_FIELD_RULES["login"]:
        assert field in summary or field in params, (
            f"login 剔除字段 {field!r} 不在返回形状与入参签名任一面（改名孤儿？）"
        )


def test_redact_uuids_masks_canonical_only():
    """canonical UUID（带连字符）→ <uuid>；mlflow run_id（32 位无连字符 hex）不中枪。"""
    fn = telemetry.redact_uuids
    assert fn("job 550e8400-e29b-41d4-a716-446655440000 done") == "job <uuid> done"
    assert fn("run 0193a2c5f8b64d7e9c2a1b3f4d5e6f70 kept") == "run 0193a2c5f8b64d7e9c2a1b3f4d5e6f70 kept"
    assert fn("<uuid> 幂等") == "<uuid> 幂等"


class _Event:
    """agentcat Event 的结构替身：钩子只触及 resource_name/parameters/response 三面。"""


def test_redact_fields_drops_login_surfaces():
    """事件级钩子：login 的 arguments 与 response 双面（structured_content + content JSON）剔除。"""
    event: Any = _Event()  # Any 缝：结构替身直传 agentcat 契约签名（pyright 不认鸭子型）
    event.resource_name = "login"
    event.parameters = {"arguments": {"force": True, "device_name": "ws-01 (Linux)"}, "session_id": "s"}
    event.response = {
        "structured_content": {"status": "ok", "user_id": "u1"},
        "content": [
            {
                "type": "text",
                "text": '{\n  "status": "ok",\n  "user_id": "u1",\n  "device_session_id": "sid",\n  "device_name": "ws-01"\n}',
            }
        ],
    }
    out = telemetry.redact_fields(event)
    assert out is event  # agentcat 契约：改入参并原样返回
    assert event.parameters["arguments"] == {"force": True}
    assert event.response["structured_content"] == {"status": "ok"}
    assert json.loads(event.response["content"][0]["text"]) == {"status": "ok"}


def test_redact_fields_drops_job_logs():
    """事件级钩子：get_platform_job_logs 的 logs 自由文本双承载面剔除，余键照常上报。"""
    event: Any = _Event()
    event.resource_name = "get_platform_job_logs"
    event.parameters = {"arguments": {"job_id": "j1", "tail_lines": 500}}
    event.response = {
        "structured_content": {
            "job_id": "j1",
            "status": "running",
            "terminal": False,
            "tail_lines": 500,
            "logs": "epoch=1 loss=0.5",
        },
        "content": [
            {
                "type": "text",
                "text": '{\n  "job_id": "j1",\n  "status": "running",\n  "terminal": false,\n  "tail_lines": 500,\n  "logs": "epoch=1 loss=0.5"\n}',
            }
        ],
    }
    out = telemetry.redact_fields(event)
    assert out is event  # agentcat 契约：改入参并原样返回
    assert event.parameters["arguments"] == {"job_id": "j1", "tail_lines": 500}  # 入参无 logs 字段，原样
    assert event.response["structured_content"] == {
        "job_id": "j1",
        "status": "running",
        "terminal": False,
        "tail_lines": 500,
    }
    assert json.loads(event.response["content"][0]["text"]) == {
        "job_id": "j1",
        "status": "running",
        "terminal": False,
        "tail_lines": 500,
    }


def test_redact_fields_non_table_tool_untouched():
    """规则表外工具：钩子零改动直通（掩码交给字符串级钩子）。"""
    event: Any = _Event()
    event.resource_name = "get_run"
    event.parameters = {"arguments": {"run_id": "r1"}}
    event.response = {"structured_content": {"run_id": "r1"}}
    snapshot = copy.deepcopy((event.parameters, event.response))
    assert telemetry.redact_fields(event) is event
    assert (event.parameters, event.response) == snapshot


def test_redact_fields_error_text_block_untouched():
    """content 文本块非 JSON（错误串）原样保留——不赌文本形态，UUID 掩码兜底。"""
    event: Any = _Event()
    event.resource_name = "login"
    event.parameters = None
    event.response = {"content": [{"type": "text", "text": "登录超时：browser flow expired"}]}
    telemetry.redact_fields(event)
    assert event.response["content"][0]["text"] == "登录超时：browser flow expired"
