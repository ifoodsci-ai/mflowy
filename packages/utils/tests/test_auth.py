"""OAuth 2.1 客户端契约：PKCE / loopback 授权流 / token 库生命周期。

缝位（预商定，见 #88）：HTTP transport = ``auth._http_json`` 单点替换；
凭据根目录 = ``auth.credentials_path()``（telemetry ``api_key_path`` 同款公开缝）。
行为只经公有接口断言；loopback 走真 127.0.0.1 socket，浏览器以桩代之。
"""

import base64
import hashlib
import json
import os
import platform
import stat
import threading
import time
import types
import urllib.parse
import urllib.request

import pytest

from mflowy.utils import auth


# ── 测试地基：JWT 桩 / 浏览器桩 / transport 缝 / 凭据缝 ─────────────────────


def _b64url(obj: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")


def _jwt(payload: dict) -> str:
    """不签名的 JWT 桩——客户端只解码展示，不验签（资源侧 gateway 统一验）。"""
    return f"{_b64url({'alg': 'RS256'})}.{_b64url(payload)}.sig"


def _use_credentials(monkeypatch, tmp_path):
    path = tmp_path / "credentials.json"
    monkeypatch.setattr(auth, "credentials_path", lambda: path)
    return path


def _fake_browser(monkeypatch, *, code: str = "c0de-1", state: str | None = None, error: str | None = None):
    """浏览器桩：捕获授权页 URL，向 loopback 回调发起真 HTTP GET（同机语义）。"""
    captured: dict[str, str] = {}

    def open_(url: str) -> bool:
        captured["url"] = url
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        params: dict[str, str] = {"state": state if state is not None else query["state"]}
        if error is not None:
            params["error"] = error
        else:
            params["code"] = code
        urllib.request.urlopen(query["redirect_uri"] + "?" + urllib.parse.urlencode(params), timeout=5).read()
        return True

    monkeypatch.setattr(auth, "webbrowser", types.SimpleNamespace(open=open_))
    return captured


def _fake_transport(monkeypatch, *, scopes_supported: list | None = None, token: dict | None = None):
    """transport 缝：记录全部出站请求；元数据/token 端点按 URL 分发。"""
    calls: list[dict] = []
    token = token or {}

    def _http_json(url, payload=None, *, method=None, timeout=30.0):
        calls.append({"url": url, "payload": payload, "method": method or ("POST" if payload is not None else "GET")})
        if url.endswith("/.well-known/oauth-authorization-server"):
            return {"issuer": "mflowy", "scopes_supported": scopes_supported, "code_challenge_methods_supported": ["S256"]}
        if url.endswith("/oauth/token"):
            if isinstance(token, Exception):
                raise token
            return token
        raise AssertionError(f"意外请求: {url}")

    monkeypatch.setattr(auth, "_http_json", _http_json)
    return calls


def _token_payload(**over) -> dict:
    now = int(time.time())
    payload = {"iss": "mflowy", "sub": "user-42", "client_id": auth.CLIENT_ID, "scope": "data:read train:run",
               "session_id": "sess-7", "consent_id": "cons-1", "exp": now + 900, "iat": now}
    payload.update(over)
    return payload


def _token_response(payload: dict | None = None, refresh: str = "rf-new") -> dict:
    return {"access_token": _jwt(payload or _token_payload()), "token_type": "Bearer", "expires_in": 899,
            "refresh_token": refresh, "scope": "data:read train:run"}


# ── login 全链（loopback 真socket + 浏览器桩 + transport 缝）─────────────────


def test_login_happy_path_authorizes_and_stores_credentials(monkeypatch, tmp_path):
    """完成浏览器授权 → 兑换 → 凭据原子落盘（0600），返回身份摘要。"""
    creds_file = _use_credentials(monkeypatch, tmp_path)
    monkeypatch.delenv(auth.ENV_BASE_URL, raising=False)
    browser = _fake_browser(monkeypatch, code="c0de-1")
    calls = _fake_transport(monkeypatch, scopes_supported=["data:read", "train:run"],
                            token=_token_response())

    result = auth.login(timeout_seconds=10)

    # 授权页 URL 形状：SPA 入口 + 全小写下划线 query + S256 + loopback redirect
    parts = urllib.parse.urlsplit(browser["url"])
    assert (parts.scheme, parts.hostname, parts.path) == ("http", "localhost", "/oauth/authorize")
    q = dict(urllib.parse.parse_qsl(parts.query))
    redirect = q["redirect_uri"]
    assert redirect.startswith("http://127.0.0.1:") and redirect.endswith("/callback")
    assert q["response_type"] == "code" and q["client_id"] == auth.CLIENT_ID
    assert q["code_challenge_method"] == "S256" and q["code_challenge"]
    assert q["scope"] == "data:read train:run" and q["state"] and q["device_name"]

    # 缺省 scope 来自元数据（客户端不拷目录）；兑换体为 JSON 契约字段
    assert any(c["url"].endswith("/.well-known/oauth-authorization-server") for c in calls)
    exchange = next(c for c in calls if c["url"].endswith("/oauth/token"))
    assert exchange["method"] == "POST"
    payload = exchange["payload"]
    assert payload["grant_type"] == "authorization_code" and payload["code"] == "c0de-1"
    assert payload["client_id"] == auth.CLIENT_ID and payload["redirect_uri"] == redirect
    assert payload["code_verifier"] and auth.pkce_challenge(payload["code_verifier"]) == q["code_challenge"]

    # 返回摘要：JWT claims 解码展示（不验签）
    assert result["status"] == "logged_in"
    assert result["base_url"] == auth.DEFAULT_BASE_URL
    assert result["user_id"] == "user-42"
    assert result["scopes"] == ["data:read", "train:run"]
    assert result["device_session_id"] == "sess-7"
    assert result["device_name"] == f"{platform.node()} ({platform.system()})"

    # 凭据落盘：单文件、仅本人可读、完整 token 语义
    stored = json.loads(creds_file.read_text())
    assert stat.S_IMODE(creds_file.stat().st_mode) == 0o600
    assert stored["base_url"] == auth.DEFAULT_BASE_URL
    assert stored["refresh_token"] == "rf-new"
    assert stored["user_id"] == "user-42" and stored["session_id"] == "sess-7"
    assert stored["scope"] == "data:read train:run"


def test_login_consent_denied_raises_login_denied(monkeypatch, tmp_path):
    """consent 页拒绝（回调 error=access_denied）→ LoginDenied，零凭据落盘。"""
    _use_credentials(monkeypatch, tmp_path)
    _fake_browser(monkeypatch, error="access_denied")
    _fake_transport(monkeypatch, scopes_supported=["data:read"])

    with pytest.raises(auth.LoginDenied, match="access_denied"):
        auth.login(timeout_seconds=10, scopes=["data:read"])
    assert not (tmp_path / "credentials.json").exists()


def test_login_state_mismatch_raises(monkeypatch, tmp_path):
    """回调 state 不符（串台/CSRF）→ StateMismatch，不兑换、零落盘。"""
    _use_credentials(monkeypatch, tmp_path)
    _fake_browser(monkeypatch, state="attacker-state")
    calls = _fake_transport(monkeypatch, scopes_supported=["data:read"])

    with pytest.raises(auth.StateMismatch):
        auth.login(timeout_seconds=10, scopes=["data:read"])
    assert not any(c["url"].endswith("/oauth/token") for c in calls)


def test_login_browser_silent_times_out(monkeypatch, tmp_path):
    """浏览器全程不回调 → TimeoutError（可重试语义；consent 记忆后二次秒过）。"""

    def open_(_url):
        return True  # "打开成功"但用户不操作

    monkeypatch.setattr(auth, "webbrowser", types.SimpleNamespace(open=open_))
    _use_credentials(monkeypatch, tmp_path)
    _fake_transport(monkeypatch, scopes_supported=["data:read"])

    with pytest.raises(TimeoutError, match="超时"):
        auth.login(timeout_seconds=1, scopes=["data:read"])


def test_login_no_browser_raises_with_manual_url(monkeypatch, tmp_path):
    """无法拉起浏览器 → RuntimeError 附手动 URL + headless 不支持提示。"""
    _use_credentials(monkeypatch, tmp_path)
    monkeypatch.setattr(auth, "webbrowser", types.SimpleNamespace(open=lambda _u: False))
    _fake_transport(monkeypatch, scopes_supported=["data:read"])

    with pytest.raises(RuntimeError, match=r"/oauth/authorize\?"):
        auth.login(timeout_seconds=5, scopes=["data:read"])
    with pytest.raises(RuntimeError, match="headless"):
        auth.login(timeout_seconds=5, scopes=["data:read"])


def test_login_explicit_scopes_skip_metadata_and_env_base_url_wins(monkeypatch, tmp_path):
    """显式 scopes 不拉元数据（少一跳）；base_url 优先级 = 参数 > env > 缺省。"""
    _use_credentials(monkeypatch, tmp_path)
    monkeypatch.setenv(auth.ENV_BASE_URL, "http://env-base:1")
    browser = _fake_browser(monkeypatch)
    calls = _fake_transport(monkeypatch, token=_token_response())

    result = auth.login(timeout_seconds=10, scopes=["data:read"], base_url="http://param-base:2")

    assert result["base_url"] == "http://param-base:2"
    assert not any(c["url"].endswith("/.well-known/oauth-authorization-server") for c in calls)
    assert browser["url"].startswith("http://param-base:2/oauth/authorize?")

    result_env = auth.login(timeout_seconds=10, scopes=["data:read"], force=True, base_url=None)
    assert result_env["base_url"] == "http://env-base:1"


# ── login 幂等（already_logged_in / force / 环境切换）────────────────────────


def _seed_credentials(monkeypatch, tmp_path, *, base_url: str | None = None, **over) -> dict:
    """落一份既有凭据（login/ensure 的输入态），返回写盘内容。"""
    _use_credentials(monkeypatch, tmp_path)
    stored = {
        "base_url": base_url or auth.DEFAULT_BASE_URL,
        "access_token": _jwt(_token_payload()),
        "refresh_token": "rf-old",
        "scope": "data:read train:run",
        "user_id": "user-42",
        "session_id": "sess-7",
        "device_name": "seed-host (Linux)",
        "obtained_at": time.time(),
        "expires_at": time.time() + 900,
    }
    stored.update(over)
    (tmp_path / "credentials.json").write_text(json.dumps(stored), encoding="utf-8")
    return stored


def test_login_idempotent_when_credentials_exist(monkeypatch, tmp_path):
    """同环境已有凭据 → already_logged_in 摘要直返：零浏览器、零网络。"""
    _seed_credentials(monkeypatch, tmp_path)

    def _boom(*_a, **_k):  # 浏览器/网络任何一跳被触发即失败
        raise AssertionError("already_logged_in 分支不应触网/弹浏览器")

    monkeypatch.setattr(auth, "webbrowser", types.SimpleNamespace(open=_boom))
    monkeypatch.setattr(auth, "_http_json", _boom)

    result = auth.login(timeout_seconds=5)
    assert result["status"] == "already_logged_in"
    assert result["user_id"] == "user-42" and result["scopes"] == ["data:read", "train:run"]


def test_login_force_reruns_flow_and_base_switch_overwrites(monkeypatch, tmp_path):
    """force=True 重走浏览器授权；切换 base_url 视为有意切换——覆写旧凭据。"""
    _seed_credentials(monkeypatch, tmp_path)  # 缺省 base 的旧凭据
    browser = _fake_browser(monkeypatch, code="c0de-2")
    _fake_transport(monkeypatch, scopes_supported=["data:read"],
                    token=_token_response(_token_payload(sub="user-99"), refresh="rf-new"))

    result = auth.login(timeout_seconds=10, base_url="http://other-base:3", force=True)

    assert result["status"] == "logged_in" and result["user_id"] == "user-99"
    assert browser["url"].startswith("http://other-base:3/oauth/authorize?")
    stored = json.loads((tmp_path / "credentials.json").read_text())
    assert stored["base_url"] == "http://other-base:3" and stored["refresh_token"] == "rf-new"


# ── PKCE（S256）──────────────────────────────────────────────────────────────


def test_pkce_challenge_matches_rfc7636_vector():
    """RFC 7636 Appendix B 官方向量：verifier → challenge 必须逐字节命中。"""
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    expected = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    assert auth.pkce_challenge(verifier) == expected


def test_pkce_challenge_is_unpadded_urlsafe():
    """challenge 无 padding、仅 urlsafe 字符集（授权服务器严格比对）。"""
    challenge = auth.pkce_challenge("a" * 43)
    assert "=" not in challenge
    assert all(c.isalnum() or c in "-_" for c in challenge)


def test_pkce_pair_verifier_length_in_rfc_range():
    """verifier 长度 ∈ [43,128]（RFC 7636 §4.1），challenge 与 verifier 一致可复算。"""
    for _ in range(5):
        verifier, challenge = auth.pkce_pair()
        assert 43 <= len(verifier) <= 128
        assert challenge == auth.pkce_challenge(verifier)
