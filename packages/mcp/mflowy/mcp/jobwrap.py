"""mflowy-job — 远程任务容器入口（#158 决议：任务包装器与 zip 往返机制）。

用法（平台 Job 的 cmd/args 提交面，RemoteJobProvider 构造）::

    mflowy-job wrap --tool hyperparameters_search --kwargs-json '<单 argv JSON>' [--inherit-from <job_id>]

exec 风格的结构性收益（三陷阱消失）：转义（单 JSON argv，不经 shell 拼接）、
geesefs 0666 无执行位（venv bin console script，无需 chmod + sh 包装）、写前
mkdir（pack 自理目录）。流程（#158/#187）：

1. 首行 stdout 回显 ``mflowy <version>``（版本可见性——S0 独立版本探针随之退役）；
2. ``--inherit-from``：解包 ``outputs/{job_id}.zip`` 进 task_dir（滚雪球合并）；
   坏 zip fail-loud 附同实验候选 zip 清单；
3. 经 CLI runner 分发面执行工具（``run_tool`` 单源；容器内恒 local 形态）；
4. **无条件 pack**：task_dir 全量 → ``outputs/{MFLOWY_JOB_ID}.zip``（tmp+replace
   原子落；pack 失败不覆写工具退出码——stderr 警告，参照调用方模式 ``pack || true``）；
5. 以工具退出码退出（``task_cmd; rc=$?; pack || true; exit $rc`` 的结构化内建）。

环境契约（镜像烧 ENV，#161）：``MFLOWY_TASK_DIR=/user-data/workspace``、
``MLFLOW_TRACKING_URI=sqlite:////user-data/workspace/mlflow.db``、
``MFLOWY_JOB_ID``（平台注入）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# outputs 挂载点（平台容器契约字面量；测试经 monkeypatch 替换）
OUTPUTS_DIR = Path("/user-data/outputs")


def main(argv: list[str] | None = None) -> None:
    from mflowy.utils.logging import setup

    setup()  # 日志绑定 stderr（stdout 只走业务数据 + 版本回显）
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "wrap":
        print(
            f"Usage: mflowy-job wrap --tool <tool> --kwargs-json <json> [--inherit-from <job_id>]（got: {argv}）",
            file=sys.stderr,
        )
        raise SystemExit(2)
    raise SystemExit(_wrap(argv[1:]))


def _wrap(args: list[str]) -> int:
    import argparse
    from importlib.metadata import PackageNotFoundError, version

    parser = argparse.ArgumentParser(prog="mflowy-job wrap")
    parser.add_argument("--tool", required=True)
    parser.add_argument("--kwargs-json", required=True)
    parser.add_argument("--inherit-from", default=None)
    parsed = parser.parse_args(args)

    # 1) 版本回显（#187：wrap 首行 stdout = mflowy <version>）
    try:
        print(f"mflowy {version('mflowy')}")
    except PackageNotFoundError:  # pragma: no cover —— workspace 源码跑法无聚合 dist
        print("mflowy <unknown-version>")

    import json

    try:
        kwargs = json.loads(parsed.kwargs_json)
    except json.JSONDecodeError as e:
        print(f"Error: --kwargs-json 非 JSON：{e}", file=sys.stderr)
        return 2
    if not isinstance(kwargs, dict):
        print(f"Error: --kwargs-json 须为 JSON 对象（got {type(kwargs).__name__}）", file=sys.stderr)
        return 2

    # 2) 继承解包（滚雪球合并进 task_dir；zip 根 = workspace 根）
    if parsed.inherit_from:
        _unpack_inherit(parsed.inherit_from)

    # 3) 执行工具（CLI runner 分发面单源——env 钉 task_dir、退出码语义同 runner）
    from .runner import run_tool

    rc = run_tool(parsed.tool, kwargs)

    # 4) 无条件 pack（失败不覆写 rc）；5) 工具 rc 退出
    _pack_outputs()
    return rc


def _task_dir() -> Path:
    from mflowy.utils.path import task_dir_or_none

    d = task_dir_or_none()
    if d is None:
        raise ValueError("task_dir 未绑定——run_tool 前置 env（MFLOWY_TASK_DIR）或锚推导缺失")
    return d


def _unpack_inherit(job_id: str) -> None:
    """解包继承源 zip 进 task_dir（合并覆写）；坏 zip/缺 zip fail-loud 附候选清单。"""
    import zipfile

    src = OUTPUTS_DIR / f"{job_id}.zip"
    candidates = sorted(p.name for p in OUTPUTS_DIR.glob("*.zip")) if OUTPUTS_DIR.is_dir() else []
    if not src.is_file():
        raise FileNotFoundError(f"继承源 zip 不存在：{src}——同实验候选：{candidates or '无'}")
    dest = _task_dir()
    try:
        with zipfile.ZipFile(src) as zf:
            zf.extractall(dest)  # 滚雪球合并：mlflow.db/.mlruns/ 前序态就地覆写
    except zipfile.BadZipFile as e:
        raise ValueError(f"继承源 zip 损坏（{src}）：{e}——同实验候选：{candidates or '无'}") from e
    print(f"inherit unpacked: {job_id}.zip -> {dest}", file=sys.stderr)


def _pack_outputs() -> None:
    """task_dir 全量 → ``outputs/{MFLOWY_JOB_ID}.zip``（tmp+replace；无条件，尽力而为）。

    pack 失败仅 stderr 警告（参照模式 ``pack || true``——工具退出码是事实源）；
    dotdir（.mlruns/.mflowy）与全部文件一概入包（#155：zip 必含追踪态整树）。
    """
    import os
    import zipfile

    job_id = os.environ.get("MFLOWY_JOB_ID", "")
    if not job_id:
        print("warning: MFLOWY_JOB_ID 未设——跳过 pack（容器契约下平台恒注入；本地直跑无 zip 归档面）", file=sys.stderr)
        return
    dest = OUTPUTS_DIR / f"{job_id}.zip"
    tmp = dest.with_name(dest.name + ".part")
    try:
        src_dir = _task_dir()  # try 内——任何失败只降级为 pack 警告，不覆写工具 rc
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(src_dir.rglob("*")):
                zf.write(p, p.relative_to(src_dir).as_posix())
        tmp.replace(dest)
        print(f"zip packed: {job_id}.zip", file=sys.stderr)
    except Exception as e:  # 尽力而为：不覆写工具 rc（#158 参照模式）
        tmp.unlink(missing_ok=True)
        print(f"warning: pack 失败（{e}）——产物未归档，继承链断裂风险", file=sys.stderr)


if __name__ == "__main__":
    main()
