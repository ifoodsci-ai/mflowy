"""OAuth 2.1 客户端契约：PKCE / loopback 授权流 / token 库生命周期。

缝位（预商定，见 #88）：HTTP transport = ``auth._http_json`` 单点替换；
凭据根目录 = ``auth.credentials_path()``（telemetry ``api_key_path`` 同款公开缝）。
行为只经公有接口断言；loopback 走真 127.0.0.1 socket，浏览器以桩代之。
"""

import base64
import hashlib
import json
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from mflowy.utils import auth


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
