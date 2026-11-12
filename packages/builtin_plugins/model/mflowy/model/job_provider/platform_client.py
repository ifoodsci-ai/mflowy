"""RemoteExperimentClient — 上游实验服务 API 的同步 HTTP 客户端（#157 决议）。

契约单源 = mflowy-platform 仓 ``apps/server/apis/*.proto`` + 网关实际行为；
消费语义见各方法 docstring。**已知上游漂移**（上游修好即删对应行）：

- 网关 OpenAPI 快照（spec.json）不带 ``/api/v1`` 前缀，实际路由带——路径以 proto
  service 绑定为准；
- spec 的 rpcStatus 错误样板网关实际不回——错误体按三形态提取（见 ``_error_message``）；
- ``GET /jobs/{id}/logs`` 的 ``tail_lines`` 上游容错 ``<=0 → 500``、上限 10000——
  我方语义（1–10000 fail-loud）由消费侧工具把关，不依赖上游容错；
- list files 的分页参数（page_token/max_keys）被上游忽略——按全量语义消费。

鉴权：逐请求 ``ensure_access_token``；401 → force 强刷一次重放一次，仍 401 →
``LoginRequiredError``。base_url 解析链与 auth 同源（``resolve_base_url``，不另立
PLATFORM var）。本模块只做 HTTP/wire 面，业务编排（同步指纹、preset 名解析、
JobOutcome）住 ensure 工具（#159）与 RemoteJobProvider（remote.py）。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

import httpx2
from mflowy.utils.auth import LoginRequiredError, ensure_access_token, resolve_base_url
from pydantic import BaseModel, ConfigDict

# 三挂载字面量（上游 S3 key 语义：files/{org}/{exp}/user-data/{mount}/…）——
# 未知值 fail-loud 附候选（红线 1），不静默透传给上游
MOUNT_TYPES = frozenset({"uploads", "workspace", "outputs"})

# Job 状态机（proto 注释）：terminal 判定归 succeeded|failed|lost|canceled
# （#159：logs 工具的 terminal 返回即此集——单源在此，消费方不另写）
JOB_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "lost", "canceled"})


class PlatformApiError(RuntimeError):
    """平台 API 调用失败（4xx/5xx/传输层）。``endpoint`` 已剥凭据（presigned 剥 query）。"""

    def __init__(self, endpoint: str, base_url: str, message: str):
        super().__init__(
            f"{endpoint} 失败：{message}（base_url={base_url}）" if base_url else f"{endpoint} 失败：{message}"
        )
        self.endpoint = endpoint
        self.base_url = base_url


class PlatformNotFoundError(PlatformApiError):
    """404——ensure「实验不存在」分支消费（绝不静默转创建）。"""


# ── wire 模型（外部边界校验即意义：pydantic + extra=ignore；时间戳 int64 原样）──
# proto3 JSON 序面：**零值字段整体省略**（gpu_count=0 的 CPU 档、size=0 的空文件……）——
# 语义上可为零的标量一律带缺省，缺字段≠形状漂移（S3 实测 gpu_count 省略案）。


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Experiment(_Wire):
    id: str
    name: str
    description: str = ""
    status: str
    created_at: int
    updated_at: int


class Job(_Wire):
    id: str
    cmd: str
    args: list[str] = []  # proto3 空数组省略
    preset_id: str
    status: str
    error_code: str = ""
    error_message: str = ""
    created_at: int = 0


class JobPreset(_Wire):
    id: str
    name: str
    version: int
    cpu_millis: int
    memory_mi: int
    gpu_count: int = 0  # proto3 零值省略——CPU 档不带该字段（S3 实测）
    active_deadline_seconds: int
    image: str
    ttl_seconds_after_finished: int
    workdir: str
    # None = HTTP 面缺席/未知（上游 #45 regen 缺失窗口；修复后字段自然出现，client 无迁移动作）
    workspace_size_mi: int | None = None


class FileEntry(_Wire):
    """list files 条目——filename 为挂载根相对路径（含目录段）。"""

    mount_type: str
    filename: str
    size: int = 0  # proto3 零值省略（空文件）
    content_type: str = ""
    etag: str = ""
    last_modified: int = 0


class FileMetadata(_Wire):
    size: int = 0
    content_type: str = ""
    etag: str = ""
    last_modified: int = 0


def _check_mount_type(mount_type: str, *, allow_empty: bool = False) -> str:
    if mount_type in MOUNT_TYPES or (allow_empty and mount_type == ""):
        return mount_type
    raise ValueError(f"未知 mount_type {mount_type!r}——可用值：{sorted(MOUNT_TYPES)}")


def _error_message(resp: httpx2.Response) -> str:
    """网关错误体三形态依次尝试（#157 §5）：``{status,message}`` proto → ``{"error","code"}`` → ``{"error"}``。"""
    try:
        body = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code}: {resp.text[:200]}"
    if isinstance(body, dict):
        if body.get("message") is not None:
            return f"HTTP {resp.status_code}: {body['message']}"
        if body.get("error") is not None:
            code = body.get("code")
            detail = f"{body['error']}（{code}）" if code else str(body["error"])
            return f"HTTP {resp.status_code}: {detail}"
    return f"HTTP {resp.status_code}: {resp.text[:200]}"


def _s3_code(resp: httpx2.Response) -> str:
    """S3 错误 XML 的 ``<Code>``（presigned 传输段错误面；只取码不回显原文——签名材料不得进消息）。"""
    m = re.search(r"<Code>([^<]+)</Code>", resp.text[:2000])
    return m.group(1) if m else "unknown"


def _presigned_endpoint(method: str, url: str) -> str:
    """``METHOD scheme://host/path``——剥 query（presigned query 含 15 分钟有效的 X-Amz-Signature）。"""
    parts = urlsplit(url)
    return f"{method} {parts.scheme}://{parts.netloc}{parts.path}"


class RemoteExperimentClient:
    """上游实验服务的同步客户端（httpx2）——九方法面（``list_jobs`` 砍除，#159）。"""

    def __init__(
        self, base_url: str | None = None, *, timeout: float = 30.0, transport: httpx2.BaseTransport | None = None
    ):
        # 与 auth 同链同缺省（构造参数 > MFLOWY_BASE_URL env > 本地 NodePort）
        self._base = resolve_base_url(base_url).rstrip("/")
        self._client = httpx2.Client(base_url=self._base, timeout=timeout, transport=transport)

    # ── 请求骨架：鉴权 + 401 强刷单次重放 + 错误族映射 ──────────────────────

    def _send(self, method: str, path: str, *, params=None, json_body=None, token: str) -> httpx2.Response:
        try:
            return self._client.request(
                method, path, params=params, json=json_body, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx2.RequestError as e:  # 传输层无 HTTP status——统一归 PlatformApiError（S8）
            raise PlatformApiError(
                f"{method} {self._base}{path}",
                self._base,
                f"平台不可达（{e.__class__.__name__}: {e}）——检查 MFLOWY_BASE_URL 指向与 NodePort 可达性",
            ) from e

    def _request(self, method: str, path: str, *, params=None, json_body=None) -> httpx2.Response:
        resp = self._send(method, path, params=params, json_body=json_body, token=ensure_access_token(self._base))
        if resp.status_code == 401:  # 本地 exp 判断不可信（时钟漂移/服务端吊销）——force 绕快路径
            resp = self._send(
                method, path, params=params, json_body=json_body, token=ensure_access_token(self._base, force=True)
            )
            if resp.status_code == 401:
                raise LoginRequiredError(
                    f"平台拒绝 access token（{method} {path} 401，强刷重放仍 401）——请重新调用 login"
                )
        return resp

    def _get(self, path: str, *, params=None):
        resp = self._request("GET", path, params=params)
        return self._json(resp)

    def _post(self, path: str, *, json_body: dict):
        resp = self._request("POST", path, json_body=json_body)
        return self._json(resp)

    def _json(self, resp: httpx2.Response) -> dict:
        # endpoint 单一渲染（404/4xx/5xx/非 JSON 四路共用）
        endpoint = f"{resp.request.method} {resp.request.url.path}"
        if resp.status_code == 404:
            # 候选清单纪律（#157 §5）：404 附已查 id（在 endpoint 路径里）+ 建议动作
            raise PlatformNotFoundError(endpoint, self._base, f"{_error_message(resp)}——确认 id 存在或先创建")
        if resp.status_code >= 400:
            raise PlatformApiError(endpoint, self._base, _error_message(resp))
        try:
            body = resp.json()
        except ValueError as e:
            raise PlatformApiError(endpoint, self._base, f"非 JSON 响应体：{resp.text[:200]}") from e
        if not isinstance(body, dict):
            raise PlatformApiError(endpoint, self._base, f"意外响应体形状：{type(body).__name__}")
        return body

    # ── 实验查建 ──────────────────────────────────────────────────────────

    def get_experiment(self, experiment_id: str) -> Experiment:
        """查实验——不存在 404 → ``PlatformNotFoundError``（ensure 消费：绝不静默转创建）。"""
        return Experiment.model_validate(self._get(f"/api/v1/experiments/{experiment_id}")["experiment"])

    def create_experiment(self, name: str, desc: str = "") -> Experiment:
        """建实验——不暴露 tags/settings（#157：调用方只关心容器本身）。"""
        return Experiment.model_validate(
            self._post("/api/v1/experiments", json_body={"name": name, "description": desc})["experiment"]
        )

    # ── 文件面（presigned 15 分钟有效期）───────────────────────────────────

    def get_upload_url(
        self, experiment_id: str, mount_type: str, filename: str, content_type: str
    ) -> tuple[str, dict[str, str]]:
        """取 presigned PUT 凭据——返回 (url, headers)。headers 须原样随 PUT 上送（content_type 等）。"""
        _check_mount_type(mount_type)
        body = self._post(
            f"/api/v1/experiments/{experiment_id}/files/upload-url",
            json_body={"mount_type": mount_type, "filename": filename, "content_type": content_type},
        )
        return body["upload_url"], dict(body.get("headers") or {})

    def get_download_url(self, experiment_id: str, mount_type: str, filename: str) -> tuple[str, FileMetadata]:
        """取 presigned GET 凭据——返回 (url, metadata)。"""
        _check_mount_type(mount_type)
        body = self._post(
            f"/api/v1/experiments/{experiment_id}/files/download-url",
            json_body={"mount_type": mount_type, "filename": filename},
        )
        return body["download_url"], FileMetadata.model_validate(body.get("metadata") or {})

    def list_files(self, experiment_id: str, mount_type: str = "", prefix: str = "") -> list[FileEntry]:
        """列实验文件——服务端目录树（FileDir 嵌套）客户端拍平；分页被上游忽略，按全量语义消费。"""
        _check_mount_type(mount_type, allow_empty=True)
        entries: list[FileEntry] = []

        def _walk(node: dict) -> None:
            entries.extend(FileEntry.model_validate(f) for f in node.get("files") or [])
            for sub in node.get("dirs") or []:
                _walk(sub)

        for root in (
            self._get(
                f"/api/v1/experiments/{experiment_id}/files", params={"mount_type": mount_type, "prefix": prefix}
            ).get("entries")
            or []
        ):
            _walk(root)
        return entries

    # ── 任务面 ────────────────────────────────────────────────────────────

    def create_job(self, cmd: str, args: list[str], experiment_id: str, preset_id: str) -> Job:
        """提交任务（preset_id 直传——名→id 解析属 provider 编排，remote.py）。"""
        body = self._post(
            "/api/v1/jobs",
            json_body={"cmd": cmd, "args": args, "experiment_id": experiment_id, "preset_id": preset_id},
        )
        return Job.model_validate(body["job"])

    def get_job(self, job_id: str) -> Job:
        return Job.model_validate(self._get(f"/api/v1/jobs/{job_id}")["job"])

    def get_job_logs(self, job_id: str, tail_lines: int) -> str:
        """一次性全量文本（非流式）——``tail_lines`` 语义由调用方保证（上游 <=0→500 是其容错非我方语义）。"""
        return self._get(f"/api/v1/jobs/{job_id}/logs", params={"tail_lines": tail_lines})["logs"]

    def list_presets(self) -> list[JobPreset]:
        """列 preset（缺省每名最新版）——provider 每次 create_job 前新鲜解析名→id（版本链 append-only）。"""
        return [JobPreset.model_validate(p) for p in self._get("/api/v1/job-presets")["presets"]]


# ── presigned 传输段（不经网关；凭据不变量 #186：任何异常/报错不含 X-Amz 签名材料）──


def upload_presigned(
    url: str, *, headers: Mapping[str, str], source: Path, transport: httpx2.BaseTransport | None = None
) -> None:
    """presigned 单段 PUT 直传（流式，不整读内存）。失败归 ``PlatformApiError``：endpoint 剥 query、message = status + S3 ``<Code>``。"""
    try:
        with httpx2.Client(transport=transport) as client, source.open("rb") as fh:
            resp = client.put(url, content=fh, headers=dict(headers))
    except httpx2.RequestError as e:
        raise PlatformApiError(
            _presigned_endpoint("PUT", url), "", f"上传传输失败（{e.__class__.__name__}: {e}）"
        ) from e
    if resp.status_code >= 400:
        raise PlatformApiError(_presigned_endpoint("PUT", url), "", f"HTTP {resp.status_code}: S3 {_s3_code(resp)}")


def download_presigned(url: str, dest: Path, *, transport: httpx2.BaseTransport | None = None) -> None:
    """presigned GET 下载落盘（tmp+rename 原子写，任何失败不落半成品）。错误面同 ``upload_presigned``。"""
    dest.parent.mkdir(parents=True, exist_ok=True)  # CSI 目录按对象物化——写前 mkdir -p（已知陷阱）
    tmp = dest.with_name(dest.name + ".part")
    try:
        with httpx2.Client(transport=transport) as client:
            with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    resp.read()  # stream 模式下错误体需显式读出再取 <Code>
                    raise PlatformApiError(
                        _presigned_endpoint("GET", url), "", f"HTTP {resp.status_code}: S3 {_s3_code(resp)}"
                    )
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_bytes():
                        fh.write(chunk)
        tmp.replace(dest)
    except httpx2.RequestError as e:
        tmp.unlink(missing_ok=True)
        raise PlatformApiError(
            _presigned_endpoint("GET", url), "", f"下载传输失败（{e.__class__.__name__}: {e}）"
        ) from e
    except BaseException:
        tmp.unlink(missing_ok=True)  # 半成品不残留（盘中途 OSError 等）
        raise
