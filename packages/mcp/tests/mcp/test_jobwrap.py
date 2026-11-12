"""mflowy-job wrap 契约测试（#158/#187 决议，切片④）。

版本回显首行 / 继承解包（滚雪球 + 坏 zip 候选清单）/ 无条件 pack（tmp+replace、
dotdir 入包、失败不覆写 rc）/ 工具退出码透传。OUTPUTS_DIR 与 MFLOWY_JOB_ID
经 monkeypatch 注入（容器契约字面量的测试缝）。
"""

import zipfile
from pathlib import Path

import pytest
from mflowy.mcp import jobwrap
from mflowy.utils.path import pin_task_dir, reset_task_dir


@pytest.fixture
def stage(tmp_path, monkeypatch):
    """容器形态替身：outputs 目录 + task_dir 钉住（image 烧 ENV 的等价注入）。"""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    task = tmp_path / "workspace"
    task.mkdir()
    monkeypatch.setattr(jobwrap, "OUTPUTS_DIR", outputs)
    monkeypatch.setenv("MFLOWY_JOB_ID", "job-t1")
    monkeypatch.setenv("MFLOWY_TASK_DIR", str(task))
    token = pin_task_dir(task)
    yield outputs, task
    reset_task_dir(token)


def _make_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


# ── 版本回显（#187）─────────────────────────────────────────────────────


def test_version_first_line(stage, capsys, monkeypatch):
    monkeypatch.setattr("mflowy.mcp.runner.run_tool", lambda tool, kwargs: 0)
    rc = jobwrap._wrap(["--tool", "list_nodes", "--kwargs-json", "{}"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out[0].startswith("mflowy ")  # 首行版本回显（S0 探针退役依据）


# ── 继承解包 ────────────────────────────────────────────────────────────


def test_unpack_inherit_merges_into_task_dir(stage):
    outputs, task = stage
    _make_zip(outputs / "job-old.zip", {"mlflow.db": b"db", ".mlruns/r1/a.parquet": b"x", "scratch/f.txt": b"f"})
    jobwrap._unpack_inherit("job-old")
    assert (task / "mlflow.db").read_bytes() == b"db"
    assert (task / ".mlruns/r1/a.parquet").exists()  # dotdir 整树随 zip
    assert (task / "scratch/f.txt").exists()


def test_unpack_inherit_missing_zip_lists_candidates(stage):
    outputs, _ = stage
    _make_zip(outputs / "job-a.zip", {"x": b"1"})
    with pytest.raises(FileNotFoundError, match=r"job-gone.*job-a\.zip"):
        jobwrap._unpack_inherit("job-gone")


def test_unpack_inherit_bad_zip_fail_loud(stage):
    outputs, _ = stage
    (outputs / "job-bad.zip").write_bytes(b"not a zip at all")
    _make_zip(outputs / "job-ok.zip", {"x": b"1"})
    with pytest.raises(ValueError, match=r"损坏.*job-ok\.zip"):
        jobwrap._unpack_inherit("job-bad")


# ── 无条件 pack ─────────────────────────────────────────────────────────


def test_pack_outputs_atomic_with_dotdirs(stage):
    outputs, task = stage
    (task / "mlflow.db").write_bytes(b"db")
    (task / ".mlruns").mkdir()
    (task / ".mlruns/keep.parquet").write_bytes(b"k")
    jobwrap._pack_outputs()
    dest = outputs / "job-t1.zip"
    assert dest.is_file() and not dest.with_name("job-t1.zip.part").exists()  # tmp+replace 无残留
    with zipfile.ZipFile(dest) as zf:
        assert set(zf.namelist()) == {"mlflow.db", ".mlruns/", ".mlruns/keep.parquet"}


def test_pack_failure_does_not_mask_tool_rc(stage, monkeypatch, capsys):
    """pack 尽力而为（参照模式 pack || true）：失败仅 stderr 警告，工具 rc 事实源。"""
    monkeypatch.setattr(jobwrap.Path, "rglob", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk gone")))
    jobwrap._pack_outputs()  # 不抛
    assert "pack 失败" in capsys.readouterr().err


def test_pack_without_job_id_skips_with_warning(stage, monkeypatch, capsys):
    monkeypatch.delenv("MFLOWY_JOB_ID")
    jobwrap._pack_outputs()
    assert "MFLOWY_JOB_ID 未设" in capsys.readouterr().err


# ── 端到端 wrap 流（run_tool 打桩）───────────────────────────────────────


def test_wrap_flow_runs_tool_then_packs(stage, capsys, monkeypatch):
    outputs, task = stage
    calls = []

    def fake_run_tool(tool, kwargs):
        calls.append((tool, kwargs))
        (task / "artifact.parquet").write_bytes(b"a")  # 工具产物落 task_dir
        return 0

    monkeypatch.setattr("mflowy.mcp.runner.run_tool", fake_run_tool)
    rc = jobwrap._wrap(["--tool", "modeling", "--kwargs-json", '{"name": "n"}'])
    assert rc == 0 and calls == [("modeling", {"name": "n"})]
    with zipfile.ZipFile(outputs / "job-t1.zip") as zf:
        assert "artifact.parquet" in zf.namelist()  # 工具产物已入包


def test_wrap_flow_tool_failure_rc_wins(stage, monkeypatch):
    outputs, task = stage
    monkeypatch.setattr("mflowy.mcp.runner.run_tool", lambda tool, kwargs: 1)
    rc = jobwrap._wrap(["--tool", "modeling", "--kwargs-json", "{}"])
    assert rc == 1
    assert (outputs / "job-t1.zip").exists()  # 失败也无条件 pack（部分产物可继承）


def test_wrap_with_inherit_end_to_end(stage, monkeypatch, capsys):
    outputs, task = stage
    _make_zip(outputs / "job-prev.zip", {"mlflow.db": b"prev-db"})
    seen = {}

    def fake_run_tool(tool, kwargs):
        seen["db"] = (task / "mlflow.db").read_bytes()  # 工具执行前继承态已就位
        return 0

    monkeypatch.setattr("mflowy.mcp.runner.run_tool", fake_run_tool)
    rc = jobwrap._wrap(["--tool", "modeling", "--kwargs-json", "{}", "--inherit-from", "job-prev"])
    assert rc == 0 and seen["db"] == b"prev-db"


def test_unpack_inherit_works_before_contextvar_pin(stage, monkeypatch):
    """S6 全链实测回归：继承解包先于 run_tool 钉锚——ContextVar 恒空，env 契约锚生效。"""
    import os

    outputs, task = stage
    monkeypatch.delenv("MFLOWY_TASK_DIR", raising=False)  # env 也缺 → cwd 缺省
    _make_zip(outputs / "job-prev.zip", {"mlflow.db": b"db"})
    monkeypatch.chdir(task)
    jobwrap._unpack_inherit("job-prev")  # 不经 ContextVar——不炸
    assert (Path.cwd() / "mlflow.db").read_bytes() == b"db"
    os.chdir("/")  # 还原（tmp cwd 已由 pytest 管理，防御性）
