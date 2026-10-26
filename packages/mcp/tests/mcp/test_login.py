"""login 平台工具契约：utils.auth 的薄包装——透传参数、直返结果、异常直通。

授权流/token 库的行为面住 ``packages/utils/tests/test_auth.py``；此层只钉
「工具面 = auth 公有面」这一件事（schema 描述约定由 test_tools_schema 钉）。
"""

from mflowy.mcp import tools
from mflowy.utils import auth


def test_login_delegates_to_auth(monkeypatch):
    """五参全量透传，返回值原样直返。"""
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"status": "logged_in", "user_id": "u"}

    monkeypatch.setattr(auth, "login", fake)
    result = tools.login(base_url="http://b:1", scopes=["data:read"], timeout_seconds=7, force=True, device_name="dev")

    assert seen == {"base_url": "http://b:1", "scopes": ["data:read"], "timeout_seconds": 7, "force": True, "device_name": "dev"}
    assert result == {"status": "logged_in", "user_id": "u"}


def test_login_defaults_flow_through(monkeypatch):
    """缺省参数（None/300/False）原样透传——缺省语义单源于 auth 签名。"""
    seen: dict = {}
    monkeypatch.setattr(auth, "login", lambda **kwargs: seen.update(kwargs) or {})
    tools.login()
    assert seen == {"base_url": None, "scopes": None, "timeout_seconds": 300, "force": False, "device_name": None}


def test_login_errors_pass_through(monkeypatch):
    """auth 的 fail-loud 异常不被包装吞掉——agent 能拿到 LoginDenied 等原语义。"""
    def fake(**_kwargs):
        raise auth.LoginDenied("用户在授权页拒绝了本次登录")

    monkeypatch.setattr(auth, "login", fake)
    try:
        tools.login()
    except auth.LoginDenied as e:
        assert "拒绝" in str(e)
    else:
        raise AssertionError("LoginDenied 应直通")
