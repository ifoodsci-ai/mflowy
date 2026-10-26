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
import secrets

from pathlib import Path

CLIENT_ID = "mflowy-cli"  # 平台预注册 public client（无 secret）
ENV_BASE_URL = "MFLOWY_BASE_URL"  # 平台地址逃生舱：生产/自建网关可被指向
DEFAULT_BASE_URL = "http://localhost:30080"  # 本地 gateway NodePort；生产经 env 覆盖

_CALLBACK_PATH = "/callback"


def pkce_challenge(verifier: str) -> str:
    """S256 challenge：BASE64URL(SHA256(verifier)) 去 padding（RFC 7636）。"""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def pkce_pair() -> tuple[str, str]:
    """(verifier, challenge)：verifier 高熵随机、长度 ∈ [43,128]。"""
    verifier = secrets.token_urlsafe(64)[:128]  # 86 字符，落在 RFC 区间
    return verifier, pkce_challenge(verifier)
