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
