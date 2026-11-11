"""ensure_platform_experiment 契约测试（#156/#157/#159 决议，切片③）。

经 ``_platform_client`` 工厂缝注入假 client（client 详测住 model 包
``test_platform_client.py``）；通道纪律经 ``_EntryChannel`` 直接绑定（布线面
由 server/runner 各自声明，本文件只测工具侧消费语义）。
"""

from types import SimpleNamespace

import pytest
from mflowy.mcp import tools
from mflowy.model.job_provider.platform_client import FileEntry, PlatformNotFoundError
from mflowy.utils.path import _EntryChannel


class FakeClient:
    """假平台 client：记录调用面，uploads 面/outputs 面按剧本回放。"""

    def __init__(self, *, uploads=(), outputs=(), jobs=(), fail_get_experiment=False):
        import itertools

        self.uploads = uploads if isinstance(uploads, dict) else {f.filename: f for f in uploads}
        self.outputs = list(outputs)
        self.jobs = {j["id"]: j for j in jobs}
        self.fail_get_experiment = fail_get_experiment
        self.calls = []
        self._ids = itertools.count(1)

    def get_experiment(self, experiment_id):
        self.calls.append(("get_experiment", experiment_id))
        if self.fail_get_experiment:
            raise PlatformNotFoundError(f"GET /api/v1/experiments/{experiment_id}", "http://gw", "HTTP 404: nope")
        return SimpleNamespace(id=experiment_id, name="e")

    def create_experiment(self, name, desc=""):
        self.calls.append(("create_experiment", name, desc))
        return SimpleNamespace(id=f"exp-new-{next(self._ids)}", name=name)

    def list_files(self, experiment_id, mount_type="", prefix=""):
        self.calls.append(("list_files", experiment_id, mount_type))
        return list(self.uploads.values()) if mount_type == "uploads" else list(self.outputs)

    def get_upload_url(self, experiment_id, mount_type, filename, content_type):
        self.calls.append(("get_upload_url", filename, content_type))
        return f"http://s3/{filename}", {"Content-Type": content_type}

    def get_job(self, job_id):
        self.calls.append(("get_job", job_id))
        return SimpleNamespace(**self.jobs[job_id])


def _entry(name, *, etag="", size=1):
    return FileEntry(mount_type="uploads", filename=name, size=size, etag=etag, last_modified=1)


@pytest.fixture
def uploaded(tmp_path, monkeypatch):
    """打 upload_presigned 桩（S3 段不属本层测试面），记录上送对。"""
    sent = []
    monkeypatch.setattr(
        "mflowy.model.job_provider.platform_client.upload_presigned",
        lambda url, *, headers, source: sent.append((url, source)),
    )
    return sent


def _file(tmp_path, name, content=b"data"):
    p = tmp_path / name
    p.write_bytes(content)
    return p


# ── 实验查建四分支 ──────────────────────────────────────────────────────


def test_create_mode_requires_name(tmp_path):
    with pytest.raises(ValueError, match="name 必填"):
        tools.ensure_platform_experiment()


def test_id_and_name_conflict_rejected():
    with pytest.raises(ValueError, match="参数来源冲突"):
        tools.ensure_platform_experiment("exp-1", name="n")


def test_id_and_desc_conflict_rejected():
    with pytest.raises(ValueError, match="参数来源冲突"):
        tools.ensure_platform_experiment("exp-1", desc="d")


def test_query_mode_never_silently_creates(monkeypatch):
    """传不存在 id → PlatformNotFoundError 直传（绝不静默转创建）。"""
    monkeypatch.setattr(tools, "_platform_client", lambda: FakeClient(fail_get_experiment=True))
    with pytest.raises(PlatformNotFoundError):
        tools.ensure_platform_experiment("exp-missing")


def test_create_mode_returns_new_id(monkeypatch):
    fake = FakeClient(outputs=[])
    monkeypatch.setattr(tools, "_platform_client", lambda: fake)
    result = tools.ensure_platform_experiment(name="我的实验", desc="d")
    assert result["platform_experiment_id"] == "exp-new-1"
    assert ("create_experiment", "我的实验", "d") in fake.calls
    assert result["files"] == {"uploaded": [], "updated": [], "skipped": []}
    assert result["inheritable"] == []


def test_query_mode_zero_side_effect(monkeypatch):
    fake = FakeClient(outputs=[])
    monkeypatch.setattr(tools, "_platform_client", lambda: fake)
    result = tools.ensure_platform_experiment("exp-7")
    assert result["platform_experiment_id"] == "exp-7"
    assert not any(c[0] == "create_experiment" for c in fake.calls)
    assert not any(c[0] == "get_upload_url" for c in fake.calls)


# ── 增量同步（MD5 ↔ etag 全等）─────────────────────────────────────────


def _md5(content: bytes) -> str:
    import hashlib

    return hashlib.md5(content).hexdigest()


def test_sync_first_upload_then_skip_then_update(tmp_path, monkeypatch, uploaded):
    a, b = _file(tmp_path, "a.csv", b"v1"), _file(tmp_path, "b.csv", b"v1")
    fake = FakeClient()
    monkeypatch.setattr(tools, "_platform_client", lambda: fake)

    # 1) 全新 → uploaded（远端名 /user-data/uploads/<basename>）
    r1 = tools.ensure_platform_experiment("exp-1", files=[str(a), str(b)])
    assert [(x["local"], x["remote"]) for x in r1["files"]["uploaded"]] == [
        (str(a), "/user-data/uploads/a.csv"),
        (str(b), "/user-data/uploads/b.csv"),
    ]
    assert len(uploaded) == 2

    # 2) 远端 etag == 本地 MD5（剥引号）→ 全部 skipped，零上送
    fake.uploads = {
        "a.csv": _entry("a.csv", etag=f'"{_md5(b"v1")}"'),
        "b.csv": _entry("b.csv", etag=_md5(b"v1")),  # 无引号裸形态同样全等
    }
    uploaded.clear()
    r2 = tools.ensure_platform_experiment("exp-1", files=[str(a), str(b)])
    assert r2["files"] == {
        "uploaded": [],
        "updated": [],
        "skipped": [
            {"local": str(a), "remote": "/user-data/uploads/a.csv"},
            {"local": str(b), "remote": "/user-data/uploads/b.csv"},
        ],
    }
    assert uploaded == []

    # 3) 改动 a → 恰 a 落 updated，b 不动
    a.write_bytes(b"v2")
    uploaded.clear()
    r3 = tools.ensure_platform_experiment("exp-1", files=[str(a), str(b)])
    assert [x["local"] for x in r3["files"]["updated"]] == [str(a)]
    assert r3["files"]["skipped"] == [{"local": str(b), "remote": "/user-data/uploads/b.csv"}]
    assert len(uploaded) == 1

    # 4) multipart etag（含 -）与 MD5 hex 永不相等 → 重传（fail-safe）
    fake.uploads["a.csv"] = _entry("a.csv", etag='"abc123-2"')
    uploaded.clear()
    r4 = tools.ensure_platform_experiment("exp-1", files=[str(a)])
    assert [x["local"] for x in r4["files"]["updated"]] == [str(a)]


def test_sync_never_deletes_unlisted_remote_files(tmp_path, monkeypatch, uploaded):
    """远端已有 c.csv 未在清单 → 不删除（无删除调用面，list 后仅对账清单内文件）。"""
    fake = FakeClient(uploads={"c.csv": _entry("c.csv", etag="zzz")})
    monkeypatch.setattr(tools, "_platform_client", lambda: fake)
    a = _file(tmp_path, "a.csv")
    tools.ensure_platform_experiment("exp-1", files=[str(a)])
    assert not any("delete" in c[0] for c in fake.calls)
    assert fake.uploads.get("c.csv") is not None


# ── 清单预检 fail-loud 家族（拒整次，不半途而废）─────────────────────────


@pytest.mark.parametrize(
    ("names", "match"),
    [((".hidden",), "dotfile"), (("sub",), "目录"), (("nope.csv",), "不存在")],
)
def test_manifest_rejections(tmp_path, monkeypatch, names, match):
    monkeypatch.setattr(tools, "_platform_client", lambda: FakeClient())
    (tmp_path / "sub").mkdir()
    files = [str(tmp_path / n) for n in names]
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        tools.ensure_platform_experiment("exp-1", files=files)


def test_manifest_basename_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_platform_client", lambda: FakeClient())
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    (d1 / "x.csv").write_bytes(b"1")
    (d2 / "x.csv").write_bytes(b"2")
    with pytest.raises(ValueError, match="撞名"):
        tools.ensure_platform_experiment("exp-1", files=[str(d1 / "x.csv"), str(d2 / "x.csv")])


def test_mcp_channel_rejects_relative_paths(monkeypatch):
    """通道纪律（#159 §6）：MCP 通道相对路径拒收 fail-loud；CLI/未知通道按 cwd 接受。"""
    monkeypatch.setattr(tools, "_platform_client", lambda: FakeClient())
    token = _EntryChannel.set("mcp")
    try:
        with pytest.raises(ValueError, match="相对路径"):
            tools.ensure_platform_experiment("exp-1", files=["rel/a.csv"])
    finally:
        _EntryChannel.reset(token)


# ── inheritable 发现面 ──────────────────────────────────────────────────


def test_inheritable_newest_first_and_unfiltered(monkeypatch):
    outputs = [_entry("j1.zip"), _entry("j2.zip"), _entry("nested/x.zip"), _entry("not-a-zip")]
    jobs = [
        {"id": "j1", "status": "succeeded", "created_at": 100},
        {"id": "j2", "status": "failed", "created_at": 200},  # 失败不过滤（部分 study db 可续跑）
    ]
    fake = FakeClient(outputs=outputs, jobs=jobs)
    monkeypatch.setattr(tools, "_platform_client", lambda: fake)
    result = tools.ensure_platform_experiment("exp-1")
    assert result["inheritable"] == [
        {"job_id": "j2", "status": "failed"},  # newest first（created_at 降序）
        {"job_id": "j1", "status": "succeeded"},
    ]
