"""RemoteExperimentClient 契约测试（#157 决议，切片②）。

全部经 ``httpx2.MockTransport`` 打桩网关/S3——零真实网络；鉴权 seam 打
``platform_client.ensure_access_token``（401 强刷重放/凭据缺失路径由 utils 侧
``test_auth.py`` 覆盖）。S8（平台不可达）与 #186 凭据不变量（presigned 报错
不含 X-Amz 签名材料）在票面上明确要求契约测试钉住。
"""

import json

import httpx2
import pytest
from mflowy.model.job_provider.platform_client import (
    FileEntry,
    Job,
    JobPreset,
    PlatformApiError,
    PlatformNotFoundError,
    RemoteExperimentClient,
    download_presigned,
    upload_presigned,
)
from mflowy.utils.auth import LoginRequiredError

GW = "http://gw.test"
EXP = {
    "id": "exp-1",
    "name": "e",
    "description": "d",
    "status": "active",
    "created_at": 1794000000,
    "updated_at": 1794000001,
}
JOB = {
    "id": "job-1",
    "cmd": "mflowy-job",
    "args": ["wrap"],
    "preset_id": "p1",
    "status": "submitted",
    "created_at": 1794000002,
}
PRESET = {
    "id": "p1",
    "name": "cpu-standard",
    "version": 2,
    "cpu_millis": 8000,
    "memory_mi": 8192,
    "gpu_count": 0,
    "active_deadline_seconds": 3600,
    "image": "training-job:0.2.0",
    "ttl_seconds_after_finished": 86400,
    "workdir": "/user-data/workspace",
    "created_at": 1794000000,
    # workspace_size_mi 故意缺席——HTTP 面 #45 regen 缺失窗口 → None
}


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    """鉴权 seam：恒发有效 token，记录 force 调用次数。"""
    calls = {"force": 0}
    from mflowy.model.job_provider import platform_client as pc

    def fake(base_url=None, *, force=False):
        if force:
            calls["force"] += 1
        return "tok-force" if force else "tok"

    monkeypatch.setattr(pc, "ensure_access_token", fake)
    return calls


def _client(handler) -> RemoteExperimentClient:
    return RemoteExperimentClient(GW, transport=httpx2.MockTransport(handler))


def _ok(payload: dict) -> httpx2.Response:
    return httpx2.Response(200, json=payload)


# ── 九方法面（wire 解析 + extra=ignore + 时间戳 int 原样）───────────────────


def test_experiment_roundtrip_and_extra_ignored():
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["Authorization"] == "Bearer tok"
        if request.method == "POST" and request.url.path == "/api/v1/experiments":
            assert json.loads(request.content) == {"name": "n", "description": "d"}
            return _ok({"status": 0, "message": "", "experiment": {**EXP, "tags": ["x"], "settings": {"a": 1}}})
        return _ok({"status": 0, "experiment": {**EXP, "org_id": "ignored"}})

    c = _client(handler)
    created = c.create_experiment("n", "d")
    assert created.id == "exp-1" and created.created_at == 1794000000  # int 原样，不做 datetime
    got = c.get_experiment("exp-1")
    assert got.name == "e" and not hasattr(got, "tags")  # extra=ignore：上游加字段不炸、不透传


def test_create_get_job_and_logs():
    state = {"logs_calls": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST" and request.url.path == "/api/v1/jobs":
            body = json.loads(request.content)
            assert body == {
                "cmd": "mflowy-job",
                "args": ["wrap", "--tool", "t"],
                "experiment_id": "exp-1",
                "preset_id": "p1",
            }
            return _ok({"status": 0, "job": JOB})
        if request.url.path == "/api/v1/jobs/job-1":
            return _ok({"status": 0, "job": {**JOB, "status": "running", "error_code": "", "error_message": ""}})
        if request.url.path == "/api/v1/jobs/job-1/logs":
            state["logs_calls"] += 1
            assert request.url.params["tail_lines"] == "500"
            return _ok({"status": 0, "logs": "== line1\n== line2"})
        raise AssertionError(f"unexpected {request.method} {request.url}")

    c = _client(handler)
    job = c.create_job("mflowy-job", ["wrap", "--tool", "t"], "exp-1", "p1")
    assert isinstance(job, Job) and job.status == "submitted" and job.args == ["wrap"]
    assert c.get_job("job-1").status == "running"
    assert c.get_job_logs("job-1", 500) == "== line1\n== line2"
    assert state["logs_calls"] == 1


def test_list_presets_workspace_size_mi_optional():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _ok({"status": 0, "presets": [PRESET, {**PRESET, "id": "p2", "workspace_size_mi": 1024}]})

    presets = _client(handler).list_presets()
    assert [type(p) for p in presets] == [JobPreset] * 2
    assert presets[0].workspace_size_mi is None  # 字段缺席 → None（缺 #45 regen 窗口语义）
    assert presets[1].workspace_size_mi == 1024


def test_list_files_flattens_server_dir_tree():
    tree = {
        "status": 0,
        "entries": [
            {
                "dirname": "user-data/uploads",
                "dirs": [
                    {
                        "dirname": "sub",
                        "dirs": [],
                        "files": [
                            {
                                "mount_type": "uploads",
                                "filename": "sub/deep.csv",
                                "size": 3,
                                "etag": '"md5b"',
                                "last_modified": 1794000003,
                            },
                        ],
                    }
                ],
                "files": [
                    {
                        "mount_type": "uploads",
                        "filename": "a.csv",
                        "size": 1,
                        "etag": '"md5a"',
                        "last_modified": 1794000000,
                    }
                ],
            }
        ],
        "next_token": "",  # 分页被上游忽略——全量语义
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.params["mount_type"] == "uploads"
        assert request.url.params["prefix"] == ""
        return _ok(tree)

    files = _client(handler).list_files("exp-1", "uploads")
    assert [f.filename for f in files] == ["a.csv", "sub/deep.csv"]
    assert isinstance(files[0], FileEntry) and files[1].etag == '"md5b"'


def test_presigned_url_methods_wire():
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/files/upload-url"):
            assert json.loads(request.content) == {
                "mount_type": "outputs",
                "filename": "j.zip",
                "content_type": "application/zip",
            }
            return _ok(
                {
                    "status": 0,
                    "upload_url": "http://s3/put?X-Amz-Signature=sekret",
                    "headers": {"Content-Type": "application/zip"},
                }
            )
        assert json.loads(request.content) == {"mount_type": "outputs", "filename": "j.zip"}
        return _ok(
            {
                "status": 0,
                "download_url": "http://s3/get?X-Amz-Signature=sekret",
                "metadata": {"size": 44, "etag": '"m"', "last_modified": 1794000004},
            }
        )

    c = _client(handler)
    url, headers = c.get_upload_url("exp-1", "outputs", "j.zip", "application/zip")
    assert url.startswith("http://s3/put") and headers == {"Content-Type": "application/zip"}
    url, meta = c.get_download_url("exp-1", "outputs", "j.zip")
    assert meta.size == 44 and meta.etag == '"m"'


def test_unknown_mount_type_fail_loud():
    c = _client(lambda request: _ok({}))
    with pytest.raises(ValueError, match=r"可用值：\['outputs', 'uploads', 'workspace'\]"):
        c.get_upload_url("exp-1", "cache", "f.bin", "application/octet-stream")


# ── 错误族 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "body", [{"status": 5, "message": "实验不存在"}, {"error": "denied", "code": "E42"}, {"error": "plain denied"}]
)
def test_error_message_three_forms(body):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(409, json=body)

    with pytest.raises(PlatformApiError) as ei:
        _client(handler).get_experiment("exp-1")
    assert "409" in str(ei.value) and ("实验不存在" in str(ei.value) or "denied" in str(ei.value))


def test_404_not_found_with_id_in_endpoint():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, json={"status": 5, "message": "not found"})

    # 候选清单纪律（#157 §5）：已查 id（endpoint 路径）+ 建议动作
    with pytest.raises(PlatformNotFoundError, match=r"exp-missing.*确认 id 存在或先创建"):
        _client(handler).get_experiment("exp-missing")


def test_non_json_body_fail_loud():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, text="<html>gateway blew up</html>")

    with pytest.raises(PlatformApiError, match="非 JSON"):
        _client(handler).list_presets()


def test_platform_unreachable_maps_to_api_error_with_hint(_token):
    """S8：死端口/不可达 → PlatformApiError，含 base_url 与可达性提示（不抛裸传输异常）。"""

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(PlatformApiError, match=r"MFLOWY_BASE_URL.*NodePort") as ei:
        _client(handler).list_presets()
    assert ei.value.base_url == GW


# ── 401 强刷单次重放（#157 §2）───────────────────────────────────────────


def test_401_force_refresh_replays_once(_token):
    seen_tokens: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen_tokens.append(request.headers["Authorization"])
        if request.headers["Authorization"] == "Bearer tok-force":
            return _ok({"status": 0, "job": JOB})
        return httpx2.Response(401, json={"error": "unauthorized"})

    job = _client(handler).get_job("job-1")
    assert job.id == "job-1"
    assert seen_tokens == ["Bearer tok", "Bearer tok-force"]  # 恰一次重放
    assert _token["force"] == 1


def test_401_after_force_still_unauthorized_requires_login(_token):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"error": "unauthorized"})

    with pytest.raises(LoginRequiredError, match="login"):
        _client(handler).get_job("job-1")
    assert _token["force"] == 1  # 强刷只做一次，不循环退避


# ── presigned 传输段（#186 凭据不变量：报错不含签名材料）───────────────────


def test_upload_presigned_s3_error_redacts_signature(tmp_path):
    """S3 403 XML 错误 → PlatformApiError；异常串不含 X-Amz/Signature，endpoint 剥 query。"""
    url = "http://s3.test/bucket/j.zip?X-Amz-Signature=SEKRET&X-Amz-Algorithm=AWS4-HMAC"
    src = tmp_path / "f.zip"
    src.write_bytes(b"PK")

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            403, text='<?xml version="1.0"?><Error><Code>AccessDenied</Code><Message>S3 拒绝</Message></Error>'
        )

    with pytest.raises(PlatformApiError) as ei:
        upload_presigned(
            url, headers={"Content-Type": "application/zip"}, source=src, transport=httpx2.MockTransport(handler)
        )
    assert ei.value.endpoint == "PUT http://s3.test/bucket/j.zip"  # query 已剥
    assert "403" in str(ei.value) and "AccessDenied" in str(ei.value)
    assert "X-Amz" not in str(ei.value) and "Signature" not in str(ei.value) and "SEKRET" not in str(ei.value)


def test_download_presigned_atomic_write_and_cleanup(tmp_path):
    url = "http://s3.test/bucket/j.zip?X-Amz-Signature=SEKRET"
    dest = tmp_path / "out" / "j.zip"

    def ok_handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=b"PK\x03\x04payload")

    download_presigned(url, dest, transport=httpx2.MockTransport(ok_handler))
    assert dest.read_bytes() == b"PK\x03\x04payload"
    assert not dest.with_name("j.zip.part").exists()

    def broken_handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, text="<Error><Code>NoSuchKey</Code></Error>")

    dest.unlink()
    with pytest.raises(PlatformApiError, match="NoSuchKey") as ei:
        download_presigned(url, dest, transport=httpx2.MockTransport(broken_handler))
    assert "X-Amz" not in str(ei.value) and "SEKRET" not in str(ei.value)  # 凭据不变量（#186）——错误分支同样不回显
    assert not dest.exists() and not dest.with_name("j.zip.part").exists()  # 失败不落半成品


def test_presigned_transport_error_redacts_signature(tmp_path):
    """传输层失败分支（非 HTTP 状态分支）同样不带签名材料——endpoint 剥 query、消息只含异常类名与串。"""
    url = "http://s3.test/bucket/j.zip?X-Amz-Signature=SEKRET"

    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    src = tmp_path / "f.bin"
    src.write_bytes(b"x")

    with pytest.raises(PlatformApiError, match="上传传输失败") as up_ei:
        upload_presigned(url, headers={}, source=src, transport=httpx2.MockTransport(refuse))
    assert up_ei.value.endpoint == "PUT http://s3.test/bucket/j.zip"
    assert "X-Amz" not in str(up_ei.value) and "SEKRET" not in str(up_ei.value)

    with pytest.raises(PlatformApiError, match="下载传输失败") as dl_ei:
        download_presigned(url, tmp_path / "j.zip", transport=httpx2.MockTransport(refuse))
    assert dl_ei.value.endpoint == "GET http://s3.test/bucket/j.zip"
    assert "X-Amz" not in str(dl_ei.value) and "SEKRET" not in str(dl_ei.value)
