"""http 数据载入器"""

import ipaddress
import json
import logging
import socket
from typing import Annotated, Any
from urllib.parse import urlparse

import pandas as pd

from mflowy.driver.handler import handler
from mflowy.middlewares.log_load_profile import log_load_profile

from . import report_loaded

logger = logging.getLogger(__name__)


@handler(log_load_profile)
def http(
    url: Annotated[str, "网络地址"],
    method: Annotated[str, "请求方法"] = "GET",
    headers: Annotated[dict[str, str] | None, "请求头"] = None,
    body: Annotated[dict[str, Any] | None, "请求体（快捷参数，支持 JSON 或表单）"] = None,
    proxy: Annotated[str, "HTTP/HTTPS 代理，如 http://user:pass@host:port"] = "",
    **kwargs,
) -> pd.DataFrame:
    """通过 HTTP/HTTPS 请求加载数据。

    url 必须为 http/https 协议，且解析后的目标 IP 不得落在私有/回环段（127/8、10/8、172.16/12、192.168/16、169.254/16、::1、fc00::/7、fe80::/10），否则抛 ValueError。默认 timeout=30s、allow_redirects=False（防 SSRF 重定向）。响应需为 JSON：顶层 list 直接成表；顶层 dict 自动探测 data/results/items/rows/records/content 中的列表，找不到则整体作为单行。

    用于"对接内部 RESTful API 取结构化数据"场景。body 在 POST/PUT/PATCH 下按 Content-Type 路由到 json 或 form；proxy 透传 http/https 代理。

    http 用"远程 API、JSON 响应"场景，csv/excel/parquet 用"本地文件"场景，python_loader 用"API 复杂、需自定义解析"场景。
    """

    import requests  # [stats] 层依赖，lazy 以免 base 环境发现机制崩溃

    # ── 参数初始化和合并 ──
    headers = headers or {}
    body = body or {}

    # 基础请求配置（优先安全设置）
    req_kwargs: dict[str, Any] = {
        "method": method.upper(),
        "url": url,
        "headers": headers.copy(),
        "timeout": kwargs.pop("timeout", 30),
        "allow_redirects": False,  # 默认禁止重定向，防止 SSRF
    }

    # 处理请求体（快捷参数 body）
    if body and req_kwargs["method"] in ("POST", "PUT", "PATCH"):
        ct = (headers.get("Content-Type", "")).lower()
        if "application/json" in ct:
            req_kwargs["json"] = body
        elif "application/x-www-form-urlencoded" in ct:
            req_kwargs["data"] = body
        else:
            req_kwargs["json"] = body  # 默认发送 JSON

    # 代理设置（快捷参数 proxy）
    if proxy:
        req_kwargs["proxies"] = {"http": proxy, "https": proxy}

    # 合并 kwargs 中支持的标准 requests 参数（覆盖上面的快捷参数）
    ALLOWED_REQUESTS_PARAMS = {
        "params",
        "data",
        "json",
        "headers",
        "cookies",
        "files",
        "auth",
        "timeout",
        "allow_redirects",
        "proxies",
        "verify",
        "stream",
        "cert",
    }
    for k in list(kwargs.keys()):
        if k in ALLOWED_REQUESTS_PARAMS:
            req_kwargs[k] = kwargs.pop(k)  # 覆盖同名字段
        # 其他参数（如 hooks 等较少用）直接保留到 kwargs，后面会记录警告

    if kwargs:
        logger.warning(f"未使用的额外参数: {list(kwargs.keys())}，已忽略")

    # ── 安全校验 ──
    _validate_url(url)

    # ── 发送请求 ──
    logger.info(f"HTTP 请求: {req_kwargs['method']} {url}")
    try:
        resp = requests.request(**req_kwargs)
    except requests.Timeout:
        raise TimeoutError(f"请求超时: {url} (timeout={req_kwargs['timeout']}s)")
    except requests.ConnectionError as e:
        raise ConnectionError(f"无法连接到服务器 {url}: {e}")
    except requests.RequestException as e:
        raise RuntimeError(f"HTTP 请求异常: {e}")

    # ── 处理响应 ──
    if resp.status_code >= 400:
        detail = resp.text[:500]  # 截取前 500 字符，避免日志过大
        logger.error(f"HTTP {resp.status_code} 错误: {detail}")
        raise RuntimeError(f"HTTP 请求返回错误状态码 {resp.status_code}，响应: {detail}")

    # 解析 JSON
    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"响应体不是有效的 JSON: {e}")

    # ── 转换为 DataFrame ──
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # 尝试常见的数据列表字段名
        candidate_keys = ["data", "results", "items", "rows", "records", "content"]
        for key in candidate_keys:
            if key in data and isinstance(data[key], list):
                df = pd.DataFrame(data[key])
                break
        else:
            # 没有找到列表，将整个字典作为一行
            df = pd.DataFrame([data])
    else:
        raise ValueError(f"不支持的 JSON 数据类型: {type(data)}，期望 list 或 dict")

    report_loaded(df)
    return df


# ── SSRF 防护配置 ─────────────────────────────────────────────
ALLOWED_PROTOCOLS = {"http", "https"}

BLOCKED_IP_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _validate_url(url: str) -> None:
    """
    验证 URL 仅使用 http/https 协议，且目标 IP 不属于内网段。
    防止 SSRF 攻击。
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_PROTOCOLS:
        raise ValueError(f"不支持的协议: {parsed.scheme}，仅允许 http/https")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"URL 中缺少有效的主机名: {url}")

    # 解析所有可能的 IP 地址（处理域名多 IP 的情况）
    try:
        addrs = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"无法解析域名 {hostname}: {e}")

    ip_set = {addr[4][0] for addr in addrs}
    for ip_str in ip_set:
        ip = ipaddress.ip_address(ip_str)
        if any(ip in network for network in BLOCKED_IP_RANGES):
            raise ValueError(f"禁止访问内部/私有网络地址: {ip} ({hostname})")

    # 也检查解析前的原始 IP 字符串（形如 http://127.0.0.1/）
    # 已由 getaddrinfo 覆盖
