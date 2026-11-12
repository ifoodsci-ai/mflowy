"""download_platform_job_artifacts + 观察面单参 tracking_uri 契约（#159 决议，切片⑤）。

download 经 `_platform_client` 缝注入假 client（presigned 段详测住
test_platform_client.py）；改写 = 纯函数契约（双形态 / 不匹配透传 / 精确与子路径）；
逐调用 client 零全局态（不污染模块级 `_client`）。
"""

import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from mflowy.mcp import tools
from mflowy.utils.path import _EntryChannel


@pytest.fixture
def download_stage(tmp_path, monkeypatch):
    """假下载段：client.download_url → 预置 zip 内容；download_presigned 打桩落盘。"""
    sink = tmp_path / "downloaded"
    sink.mkdir()
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()

    def fake_download(url, dest, *, transport=None):
        # url 形如 s3://…/{job_id}.zip——从假 client 注入的映射取内容
        job_id = url.rsplit("/", 1)[-1][: -len(".zip")]
        src = payload_dir / f"{job_id}.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        sink.joinpath(f"{job_id}.txt").write_text(url)

    monkeypatch.setattr("mflowy.model.job_provider.platform_client.download_presigned", fake_download)

    class FakeClient:
        def get_download_url(self, experiment_id, mount_type, filename):
            assert mount_type == "outputs"
            return f"s3://bucket/{filename}", SimpleNamespace(size=1)

    monkeypatch.setattr(tools, "_platform_client", lambda: FakeClient())
    return payload_dir


def _make_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def test_download_lands_target_job_dir(tmp_path, download_stage):
    _make_zip(download_stage / "job-9.zip", {"mlflow.db": b"db", ".mlruns/a/x.parquet": b"x", "sub/f.txt": b"f"})
    result = tools.download_platform_job_artifacts("job-9", "exp-1", str(tmp_path))
    dest = tmp_path / "job-9"
    assert result["local_task_dir"] == str(dest)
    assert (dest / "mlflow.db").read_bytes() == b"db" and (dest / ".mlruns/a/x.parquet").exists()
    assert result["file_count"] == 3  # 目录项不计
    assert result["mlflow_db"] == str(dest / "mlflow.db")
    assert not list(tmp_path.glob("*.zip")) and not list(tmp_path.glob("*.part"))  # zip 即弃无残留


def test_download_replaces_atomically(tmp_path, download_stage):
    _make_zip(download_stage / "job-9.zip", {"v.txt": b"old"})
    tools.download_platform_job_artifacts("job-9", "exp-1", str(tmp_path))
    _make_zip(download_stage / "job-9.zip", {"v.txt": b"new", "extra.txt": b"e"})
    result = tools.download_platform_job_artifacts("job-9", "exp-1", str(tmp_path))
    dest = tmp_path / "job-9"
    assert (dest / "v.txt").read_bytes() == b"new" and (dest / "extra.txt").exists()  # 内容最新
    assert result["file_count"] == 2  # 无旧文件残留（换位而非合并）
    assert not (tmp_path / ".job-9.replacing").exists() and not (tmp_path / ".job-9.retired").exists()


def test_download_mlflow_db_null_when_absent(tmp_path, download_stage):
    _make_zip(download_stage / "job-9.zip", {"a.txt": b"1"})
    result = tools.download_platform_job_artifacts("job-9", "exp-1", str(tmp_path))
    assert result["mlflow_db"] is None


def test_download_target_dir_missing_fail_loud(tmp_path, download_stage):
    with pytest.raises((FileNotFoundError, ValueError), match="target_dir"):
        tools.download_platform_job_artifacts("job-9", "exp-1", str(tmp_path / "nope"))


def test_download_mcp_channel_rejects_relative_target(tmp_path, download_stage):
    token = _EntryChannel.set("mcp")
    try:
        with pytest.raises(ValueError, match="相对路径"):
            tools.download_platform_job_artifacts("job-9", "exp-1", "rel/dir")
    finally:
        _EntryChannel.reset(token)


def test_download_bad_zip_fail_loud(tmp_path, download_stage):
    download_stage.joinpath("job-9.zip").write_bytes(b"garbage")
    with pytest.raises(ValueError, match="损坏"):
        tools.download_platform_job_artifacts("job-9", "exp-1", str(tmp_path))


# ── 改写纯函数（#159 §4 / #160 单参推导终稿）─────────────────────────────


class TestRewriteRemoteArtifact:
    ROOT = Path("/local/dl/job-9")

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("file:///user-data/workspace/.mlruns", "/local/dl/job-9/.mlruns"),  # 精确（file:// 形态）
            ("/user-data/workspace/.mlruns", "/local/dl/job-9/.mlruns"),  # 裸路径形态
            (
                "file:///user-data/workspace/.mlruns/r1/artifacts/x.parquet",
                "/local/dl/job-9/.mlruns/r1/artifacts/x.parquet",
            ),
            ("file:///user-data/workspace/models/m-uuid", "/local/dl/job-9/models/m-uuid"),  # logged model 双形态
            ("file:///home/other/.mlruns", "file:///home/other/.mlruns"),  # 不匹配前缀原样透传
            ("http://remote:5000/r1", "http://remote:5000/r1"),
        ],
    )
    def test_rules(self, value, expected):
        assert tools._rewrite_remote_artifact(value, self.ROOT) == expected

    def test_none_root_passthrough(self):
        assert (
            tools._rewrite_remote_artifact("file:///user-data/workspace/.mlruns", None)
            == "file:///user-data/workspace/.mlruns"
        )

    def test_prefix_not_partial_match(self):
        # /user-data/workspace-evil 不得误命中（精确 or 子路径才算）
        assert (
            tools._rewrite_remote_artifact("file:///user-data/workspace-evil/x", self.ROOT)
            == "file:///user-data/workspace-evil/x"
        )


# ── 观察面单参：逐调用 client（零全局态）─────────────────────────────────


def test_tracking_client_per_call_no_global_pollution(tmp_path, monkeypatch):
    """给定 tracking_uri → 逐调用 MlflowClient(tracking_uri=...)，模块级 _client 不动。"""
    db = tmp_path / "remote" / "mlflow.db"
    db.parent.mkdir(parents=True)
    tools._client = None
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/default.db")
    try:
        client, root = tools._tracking_client(f"sqlite:///{db}")
        assert root == db.parent  # db 目录即工件根（#155 同根契约）
        # 逐调用 client 真连该库：建实验可查（真跑 sqlite，非 mock——观察面是接缝用例）
        exp_id = client.create_experiment("tracking-remote-probe")
        assert client.get_experiment(exp_id).name == "tracking-remote-probe"
        # 全局默认库未被写入该实验
        default_client, default_root = tools._tracking_client(None)
        assert default_root is None
    finally:
        tools._client = None
