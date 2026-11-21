"""遥测字段规则表镜像契约：表键 ⊆ 已注册工具、字段 ⊆ 工具可见面、动作 ⊆ 已实现集。

剔除行是隐私契约（PRIVACY.md「不采集」按构造保证的组成件）——指向已改名/删除的
工具或字段即静默失效的脱敏，双向差集在此 fail-loud（红线 1/7）。
"""

import inspect
from importlib.metadata import entry_points

from mflowy.app import telemetry


def _registered_tools() -> set[str]:
    """已注册工具全集 = mflowy.tools entry points（CLI 分发同一真源）。"""
    return {ep.name for ep in entry_points(group="mflowy.tools")}


def _load_tool(name: str):
    """按名加载已注册工具函数（entry points 真源，与 CLI 分发同一加载面）。"""
    return next(ep.load() for ep in entry_points(group="mflowy.tools") if ep.name == name)


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
    login_fn = _load_tool("login")
    params = set(inspect.signature(login_fn).parameters)
    for field, _ in telemetry.TELEMETRY_FIELD_RULES["login"]:
        assert field in summary or field in params, (
            f"login 剔除字段 {field!r} 不在返回形状与入参签名任一面（改名孤儿？）"
        )


def test_get_platform_job_logs_rule_fields_in_return_shape(monkeypatch):
    """剔除字段必须咬合 get_platform_job_logs 返回面，否则改名孤儿静默失效。

    返回面单源 = 函数真实返回（裸 dict）——经 ``_platform_client`` 工厂缝（#159 §8）
    替假 client 取键，不取 docstring（文档可漂移）。
    """
    from mflowy.app import tools

    class _Job:
        status = "running"

    class _StubClient:
        def get_job(self, job_id):
            return _Job()

        def get_job_logs(self, job_id, tail_lines):
            return ""

    monkeypatch.setattr(tools, "_platform_client", lambda: _StubClient())
    fn = _load_tool("get_platform_job_logs")
    result = fn(job_id="j1")
    for field, _ in telemetry.TELEMETRY_FIELD_RULES["get_platform_job_logs"]:
        assert field in result, f"get_platform_job_logs 剔除字段 {field!r} 不在返回形状（改名孤儿？）"


def test_redact_uuids_masks_canonical_only():
    """canonical UUID（带连字符）→ <uuid>；mlflow run_id（32 位无连字符 hex）不中枪。"""
    fn = telemetry.redact_uuids
    assert fn("job 550e8400-e29b-41d4-a716-446655440000 done") == "job <uuid> done"
    assert fn("run 0193a2c5f8b64d7e9c2a1b3f4d5e6f70 kept") == "run 0193a2c5f8b64d7e9c2a1b3f4d5e6f70 kept"
    assert fn("<uuid> 幂等") == "<uuid> 幂等"


def test_drop_fields_applies_to_both_input_and_result():
    """事件级钩子（build_event_attrs 内嵌）：入参与返回值双面剔除规则字段。"""
    import json

    attrs = {
        a["key"]: next(iter(a["value"].values()))
        for a in telemetry.build_event_attrs(
            "get_platform_job_logs",
            {"job_id": "j1", "tail_lines": 500},
            result={
                "job_id": "j1",
                "status": "running",
                "terminal": False,
                "tail_lines": 500,
                "logs": "epoch=1 loss=0.5",
            },
        )
    }
    assert json.loads(attrs["mflowy.parameters"]) == {"job_id": "j1", "tail_lines": 500}
    response = json.loads(attrs["mflowy.response"])
    assert response == {"job_id": "j1", "status": "running", "terminal": False, "tail_lines": 500}


def test_unknown_redact_action_fail_closed():
    """规则表出现未实现动作 → 构建期 raise（整事件丢弃，绝不发原文）。"""
    from unittest.mock import patch

    with patch.dict(telemetry.TELEMETRY_FIELD_RULES, {"t": [("x", "shrink")]}):
        try:
            telemetry.build_event_attrs("t", {"x": 1})
        except ValueError as e:
            assert "未实现的遥测剔除动作" in str(e)
        else:
            raise AssertionError("fail-closed 未生效")


def test_redacted_argument_keys_cover_session_anchor():
    """#242/#243 随参数正名跟进：会话锚 observation_json（状态文件绝对路径）不进
    遥测参数——旧键 session_id 并防（存量调用方）。"""
    assert {"observation_json", "session_id"} <= set(telemetry._REDACTED_ARGUMENT_KEYS)
