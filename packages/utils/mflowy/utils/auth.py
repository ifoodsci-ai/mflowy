"""平台身份 OAuth 2.1 客户端——PKCE 授权流、loopback 回调、token 库。

契约快照见平台仓 ``docs/OAUTH_CLI.md``（2026-10-26）：授权码 + PKCE S256 +
loopback 单通道；token 端点走 **JSON 体**（非 RFC 6749 form）；refresh 轮换制，
复用旧值会被服务端判定失窃并全端下线——因此轮换必须整文件原子替换（单源原语
``write_text_atomic``），且刷新/落盘临界区持文件锁（同机多 agent host 并发是
现实场景）。

消费方：mcp 层 ``login`` 平台工具（显式授权触发）与未来远程 JobProvider
（``ensure_access_token`` / ``LoginRequiredError`` 接线点）。零第三方依赖。

测试缝（预商定）：``_http_json``（HTTP transport 单点）、``credentials_path``
（凭据根目录，telemetry ``api_key_path`` 同款）。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import platform
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from mflowy.utils.file import write_text_atomic

logger = logging.getLogger(__name__)

CLIENT_ID = "mflowy-cli"  # 平台预注册 public client（无 secret）
ENV_BASE_URL = "MFLOWY_BASE_URL"  # 平台地址逃生舱：生产/自建网关可被指向
DEFAULT_BASE_URL = "http://localhost:30080"  # 本地 gateway NodePort；生产经 env 覆盖

_CALLBACK_PATH = "/callback"
_LOCK_TIMEOUT_SECONDS = 10.0  # 凭据锁竞争上限：另一进程刷新卡死时不无限等待
_EXPIRY_SKEW_SECONDS = 30.0  # access 视为过期的提前量（时钟漂移/请求在途）


class LoginDenied(RuntimeError):
    """用户在授权页拒绝了本次登录（error=access_denied）。"""


class StateMismatch(RuntimeError):
    """loopback 回调 state 不符——疑似回调串台/CSRF。"""


class OAuthHttpError(RuntimeError):
    """OAuth 端点非 2xx：携带 HTTP 状态与 RFC 6749 error 码。"""

    def __init__(self, url: str, status: int, error: str, description: str):
        super().__init__(f"OAuth 端点 {url} 返回 {status}: {error}（{description}）")
        self.url, self.status, self.error, self.description = url, status, error, description


class LoginRequiredError(RuntimeError):
    """本地凭据缺失/失效——消费方应引导用户重新调用 login。"""


# ── PKCE（S256）──────────────────────────────────────────────────────────────


def pkce_challenge(verifier: str) -> str:
    """S256 challenge：BASE64URL(SHA256(verifier)) 去 padding（RFC 7636）。"""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def pkce_pair() -> tuple[str, str]:
    """(verifier, challenge)：verifier 高熵随机、长度 ∈ [43,128]。"""
    verifier = secrets.token_urlsafe(64)[:128]  # 86 字符，落在 RFC 区间
    return verifier, pkce_challenge(verifier)


# ── 凭据库（单文件，原子整替）───────────────────────────────────────────────


def credentials_path() -> Path:
    return Path.home() / ".mflowy" / "credentials.json"


@dataclass
class Credentials:
    """凭据单形状：登录/轮换/展示共用，整文件原子替换。"""

    base_url: str
    access_token: str
    refresh_token: str
    scope: str
    user_id: str | None
    session_id: str | None
    device_name: str | None
    obtained_at: float
    expires_at: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Credentials:
        return cls(**{f: d[f] for f in cls.__dataclass_fields__ if f in d})

    def summary(self, status: str) -> dict:
        """login 返回摘要：JWT claims 解码展示（不验签）。"""
        return {
            "status": status,
            "base_url": self.base_url,
            "user_id": self.user_id,
            "scopes": self.scope.split(),
            "access_token_expires_at": datetime.fromtimestamp(self.expires_at, timezone.utc).isoformat(),
            "device_session_id": self.session_id,
            "device_name": self.device_name,
            "credentials_path": str(credentials_path()),
        }


def jwt_payload(token: str) -> dict:
    """解码 JWT payload 段（仅展示用，不验签——资源侧统一验）。"""
    part = token.split(".")[1]
    part += "=" * (-len(part) % 4)
    return json.loads(base64.urlsafe_b64decode(part))


def _load() -> Credentials | None:
    path = credentials_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return Credentials.from_dict(data)


def _store(creds: Credentials) -> None:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    # umask 收紧后再 tmp+rename：tmp 文件自诞生即 0600，token 不经短暂 644 窗口
    old_umask = os.umask(0o177)
    try:
        write_text_atomic(path, json.dumps(asdict(creds), ensure_ascii=False, indent=2))
    finally:
        os.umask(old_umask)
    path.chmod(0o600)  # 幂等纠偏（外部改动过权限时拉回）


def _clear() -> None:
    credentials_path().unlink(missing_ok=True)


if sys.platform == "win32":
    import msvcrt

    def _lock_acquire(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _lock_release(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock_acquire(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _lock_release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def _credentials_lock(timeout: float = _LOCK_TIMEOUT_SECONDS):
    """刷新/落盘临界区互斥：轮换竞态会触发服务端复用检测（全端下线）。"""
    lock_path = credentials_path().with_name(credentials_path().name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                _lock_acquire(fd)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"凭据锁竞争超时（>{timeout:g}s）：另一 mflowy 进程正持有锁")
                time.sleep(0.05)
        yield
    finally:
        try:
            _lock_release(fd)
        finally:
            os.close(fd)


# ── HTTP transport（JSON 体；测试缝 = 模块级单点）────────────────────────────


def _http_json(url: str, payload: dict | None = None, *, method: str | None = None, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"  # 契约：JSON 体，非 RFC 6749 form
    req = urllib.request.Request(url, data=data, headers=headers, method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {}
        raise OAuthHttpError(url, e.code, str(parsed.get("error", "unknown")), str(parsed.get("error_description", body[:200]))) from e


def _resolve_base_url(base_url: str | None) -> str:
    return base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL


def _resolve_scope(base_url: str, scopes: list[str] | str | None) -> str:
    """缺省拉服务端 scope 目录全集（单源：目录真源在服务端）；显式传参原样上送。"""
    if scopes:
        return scopes if isinstance(scopes, str) else " ".join(scopes)
    meta = _http_json(f"{base_url}/.well-known/oauth-authorization-server")
    supported = meta.get("scopes_supported")
    if not supported:
        raise RuntimeError(f"{base_url} 元数据缺少 scopes_supported——无法确定缺省 scope，请显式传 scopes")
    return " ".join(supported)


def _token_request(base_url: str, payload: dict) -> dict:
    return _http_json(f"{base_url}/oauth/token", payload)


def _callback_handler(state: str, result: dict, done: threading.Event) -> type[BaseHTTPRequestHandler]:
    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path != _CALLBACK_PATH:
                self.send_error(404)
                return
            query = dict(urllib.parse.parse_qsl(parsed.query))
            if query.get("error"):
                result["error"] = query["error"]
            elif query.get("state") == state and query.get("code"):
                result["code"] = query["code"]
            else:
                result["state_ok"] = False  # state 不符或既无 code 又无 error
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            ok = "code" in result
            self.wfile.write(("mflowy 登录成功，可关闭此页。" if ok else "mflowy 登录未完成，可关闭此页后重试。").encode("utf-8"))
            done.set()

        def log_message(self, *_args):  # 回调页非诊断面，静默
            pass

    return CallbackHandler


def login(
    *,
    base_url: str | None = None,
    scopes: list[str] | str | None = None,
    timeout_seconds: int = 300,
    force: bool = False,
    device_name: str | None = None,
) -> dict:
    """登录 mflowy 平台：PKCE + 本机浏览器授权 + loopback 回调 + 凭据原子落盘。

    幂等：本地已有同环境凭据（refresh 在）→ 直接返回 ``already_logged_in`` 摘要，
    不弹浏览器；``force=True`` 强制重走授权。返回 ``logged_in``/``already_logged_in``
    身份摘要（user_id/scopes/过期时间/设备名/凭据路径）。

    失败面（fail-loud）：``LoginDenied``（consent 拒绝）/ ``StateMismatch``（回调
    串台）/ ``TimeoutError``（浏览器流程超时）/ ``OAuthHttpError``（端点非 2xx）/
    ``RuntimeError``（无法拉起浏览器——附手动 URL，首版需本机浏览器）。
    """
    base = _resolve_base_url(base_url)
    if not force:
        existing = _load()
        if existing is not None and existing.base_url == base and existing.refresh_token:
            return existing.summary("already_logged_in")

    scope = _resolve_scope(base, scopes)
    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(32)
    device = device_name or f"{platform.node()} ({platform.system()})"

    result: dict[str, Any] = {}
    done = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _callback_handler(state, result, done))
    port = server.server_address[1]
    redirect_uri = f"http://127.0.0.1:{port}{_CALLBACK_PATH}"
    authorize_url = (
        f"{base}/oauth/authorize?" + urllib.parse.urlencode({
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "device_name": device,
        })
    )

    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        if not webbrowser.open(authorize_url):
            raise RuntimeError(
                f"无法拉起本机浏览器。请在同一台机器手动打开：\n  {authorize_url}\n"
                "（首版登录需本机浏览器；headless/SSH 场景暂不支持，后续提供 PAT 通道）"
            )
        if not done.wait(timeout_seconds):
            raise TimeoutError(f"等待浏览器完成授权超时（{timeout_seconds}s）：请重试 login；已记住的同意会直接通过")
    finally:
        server.shutdown()
        server.server_close()

    if result.get("error"):
        raise LoginDenied(f"用户在授权页拒绝了本次登录（error={result['error']}）")
    if "code" not in result:
        raise StateMismatch("loopback 回调 state 不符——疑似回调串台，请重试 login")

    token = _token_request(base, {
        "grant_type": "authorization_code",
        "code": result["code"],
        "code_verifier": verifier,
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
    })
    claims = jwt_payload(token["access_token"])
    creds = Credentials(
        base_url=base,
        access_token=token["access_token"],
        refresh_token=token["refresh_token"],
        scope=token.get("scope", scope),
        user_id=claims.get("sub"),
        session_id=claims.get("session_id"),
        device_name=device,
        obtained_at=time.time(),
        expires_at=float(claims.get("exp", time.time() + token.get("expires_in", 900))),
    )
    with _credentials_lock():
        _store(creds)
    logger.info("login 完成：user_id=%s scope=%s", creds.user_id, creds.scope)
    return creds.summary("logged_in")


def ensure_access_token(base_url: str | None = None) -> str:
    """取效 access token：未过期直返；过期 → 持锁 → 双检 → refresh grant。

    refresh 为轮换制（每次返回新 refresh token，旧值立即作废）——整文件原子替换，
    且临界区持文件锁：并发二次提交旧值会触发服务端复用检测（账号全端下线）。

    失败面：无凭据 / 环境不符 / refresh 失效 → ``LoginRequiredError``（invalid_grant
    时已清本地凭据，文案含全端下线风险提示）；端点其他错误 → ``OAuthHttpError``。
    """
    creds = _load()
    if creds is None:
        raise LoginRequiredError("未登录：请先调用 login 工具完成平台授权")
    if base_url is not None and base_url != creds.base_url:
        raise LoginRequiredError(f"本地凭据属于 {creds.base_url}，与请求环境 {base_url} 不符——请用对应环境重新 login")
    if time.time() < creds.expires_at - _EXPIRY_SKEW_SECONDS:
        return creds.access_token

    with _credentials_lock():
        creds = _load()  # 双检：等锁期间他人已刷新则直接复用
        if creds is None:
            raise LoginRequiredError("未登录：请先调用 login 工具完成平台授权")
        if time.time() < creds.expires_at - _EXPIRY_SKEW_SECONDS:
            return creds.access_token
        try:
            token = _token_request(creds.base_url, {
                "grant_type": "refresh_token",
                "refresh_token": creds.refresh_token,
                "client_id": CLIENT_ID,
            })
        except OAuthHttpError as e:
            if e.error == "invalid_grant":
                _clear()
                raise LoginRequiredError(
                    "登录凭据已失效（invalid_grant），本地凭据已清除，请重新调用 login。"
                    "若非本人操作：refresh 复用会触发平台复用检测导致账号全端下线，"
                    "请到平台 /account 检查已登录设备列表。"
                ) from e
            raise
        claims = jwt_payload(token["access_token"])
        rotated = Credentials(
            base_url=creds.base_url,
            access_token=token["access_token"],
            refresh_token=token["refresh_token"],
            scope=token.get("scope", creds.scope),
            user_id=claims.get("sub", creds.user_id),
            session_id=claims.get("session_id", creds.session_id),
            device_name=creds.device_name,
            obtained_at=time.time(),
            expires_at=float(claims.get("exp", time.time() + token.get("expires_in", 900))),
        )
        _store(rotated)
        logger.info("token 已轮换：user_id=%s", rotated.user_id)
        return rotated.access_token
