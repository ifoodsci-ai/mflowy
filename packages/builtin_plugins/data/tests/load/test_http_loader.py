"""http 加载器 SSRF 防护契约：协议白名单 + 私有/回环段封锁 + 重定向恒禁。

安全门测试全部网络无关：私有/回环/链路本地用字面量 IP（getaddrinfo 本地解析），
公网与域名场景 monkeypatch 解析结果与 requests.request——测试不依赖外网。
"""

from types import SimpleNamespace

import pytest
from mflowy.data.load import http_loader
from mflowy.data.load.http_loader import _validate_url

# ── _validate_url：协议与 IP 段门 ──────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/api",  # 回环 v4
        "http://10.0.0.5/api",  # RFC1918 A 段
        "http://172.16.3.4/api",  # RFC1918 B 段
        "http://192.168.1.10/api",  # RFC1918 C 段
        "http://169.254.169.254/latest",  # 链路本地（云元数据端点）
        "http://[::1]/api",  # 回环 v6
    ],
)
def test_private_targets_rejected_offline(url):
    """字面量私有/回环/链路本地地址一律拒收（getaddrinfo 本地解析，零网络）。"""
    with pytest.raises(ValueError, match="内部/私有网络地址"):
        _validate_url(url)


@pytest.mark.parametrize("scheme", ["file", "ftp", "gopher"])
def test_non_http_schemes_rejected(scheme):
    """协议白名单仅 http/https。"""
    with pytest.raises(ValueError, match="不支持的协议"):
        _validate_url(f"{scheme}://example.com/data")


def test_missing_hostname_rejected():
    with pytest.raises(ValueError, match="主机名"):
        _validate_url("http:///path/only")


def test_unresolvable_domain_rejected(monkeypatch):
    def _gaierror(*args, **kwargs):
        raise http_loader.socket.gaierror("Name or service not known")

    monkeypatch.setattr(http_loader.socket, "getaddrinfo", _gaierror)
    with pytest.raises(ValueError, match="无法解析域名"):
        _validate_url("https://no-such.invalid/data")


def test_multi_a_record_any_private_rejects(monkeypatch):
    """多 A 记录任一落私有段即拒收（any 语义，防单记录绕过）。"""
    monkeypatch.setattr(
        http_loader.socket,
        "getaddrinfo",
        lambda h, p: [(0, 0, 0, "", ("93.184.216.34", 0)), (0, 0, 0, "", ("10.1.2.3", 0))],
    )
    with pytest.raises(ValueError, match=r"10.1.2.3"):
        _validate_url("https://multi.example.com/data")


def test_all_public_records_pass(monkeypatch):
    monkeypatch.setattr(
        http_loader.socket,
        "getaddrinfo",
        lambda h, p: [(0, 0, 0, "", ("93.184.216.34", 0))],
    )
    _validate_url("https://public.example.com/data")  # 不 raise 即通过


# ── http()：请求路径上的安全默认 ───────────────────────────────


def _fake_response(payload):
    return SimpleNamespace(status_code=200, text="", json=lambda: payload)


def test_request_defaults_and_redirect_not_reopenable(monkeypatch):
    """allow_redirects 恒 False——kwargs 携带 allow_redirects=True 也不重开（SSRF 重定向门）。"""
    monkeypatch.setattr(
        http_loader.socket,
        "getaddrinfo",
        lambda h, p: [(0, 0, 0, "", ("93.184.216.34", 0))],
    )
    captured = {}

    def _fake_request(**kwargs):
        captured.update(kwargs)
        return _fake_response({"data": [{"a": 1}, {"a": 2}]})

    import requests

    monkeypatch.setattr(requests, "request", _fake_request)

    df = http_loader.http(url="https://api.example.com/x", allow_redirects=True)

    assert captured["allow_redirects"] is False
    assert captured["timeout"] == 30
    assert list(df["a"]) == [1, 2]  # dict.data 列表字段成表


def test_private_url_never_reaches_request(monkeypatch):
    """校验先于请求：私有地址在发出任何网络调用前拒收。"""

    def _must_not_call(**kwargs):
        raise AssertionError("私有地址不得发起请求")

    import requests

    monkeypatch.setattr(requests, "request", _must_not_call)
    with pytest.raises(ValueError, match="内部/私有网络地址"):
        http_loader.http(url="http://169.254.169.254/latest/meta-data")
