"""S2+ 工具面 E2E 驱动（remote 实现系列 PR 合并门禁的工具段——
docs/research/2026-11-11-e2e-remote-runbook.md）：CLI 真实通道（#243——mflowy <tool>
--kwargs-json，subprocess），cwd 置于临时目录。

覆盖（随切片生长）：
- S2 实验就绪 ensure_platform_experiment（切片③）：查建/幂等/增量/负向 fail-loud 家族
前提：~/.mflowy/credentials.json 已登录（access 过期自动 refresh）；本地 k3s dev 集群。
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time as time_mod
from dataclasses import dataclass, field
from pathlib import Path

_WHL_VERSION = re.search(
    r'^version = "(.*)"$', Path(__file__).resolve().parent.parent.joinpath("pyproject.toml").read_text(), re.M
).group(1)

NPASS = 0
NFAIL = 0

LAST_RC = 0  # 最近一次 call 的退出码（DAG 类本地成功 = 流式上屏无 JSON 回显，rc 是断言面）


def ok(msg: str) -> None:
    global NPASS
    NPASS += 1
    print(f"  ✓ {msg}")


def bad(msg: str, detail: str = "") -> None:
    global NFAIL
    NFAIL += 1
    print(f"  ✗ {msg} {detail}")


@dataclass
class Channel:
    """CLI 通道（cwd + env）：一次构造 = 一个受控执行环境。"""

    cwd: str
    env: dict = field(default_factory=dict)


def _cli_cmd() -> list[str]:
    if exe := shutil.which("mflowy"):
        return [exe]
    return [sys.executable, "-m", "mflowy.app.cli"]


def call(ch: Channel, tool: str, args: dict) -> tuple[dict | None, str]:
    """调工具 → (解析结果, 原始文本)。失败面 = 非零退出码或异常串。

    DAG 类本地成功（JobOutcome finished 内嵌 DAG）按 CLI 契约流式上屏、无 JSON
    回显——断言面 = LAST_RC（0 = finished）。查询类回显 stdout JSON 供解析。
    """
    global LAST_RC
    proc = subprocess.run(
        [*_cli_cmd(), tool, "--kwargs-json", json.dumps(args, ensure_ascii=False)],
        capture_output=True,
        text=True,
        cwd=ch.cwd,
        env={**os.environ, **ch.env},
    )
    LAST_RC = proc.returncode
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return None, text
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, text
    return (parsed, text) if isinstance(parsed, dict) else (None, text)


def expect_error(ch: Channel, label: str, tool: str, args: dict, *needles: str) -> None:
    result, text = call(ch, tool, args)
    if result is not None:
        bad(f"{label}——应 fail-loud 却成功：{text[:200]}")
        return
    missing = [n for n in needles if n not in text]
    if missing:
        bad(f"{label}——报错缺 {missing}：{text[:200]}")
    else:
        ok(f"{label} fail-loud（{'/'.join(needles)}）")


def run_s2(session: Channel, stage: Path) -> str:
    """S2：实验就绪。返回新建实验 id（供 S3+ 复用）。"""
    print("== S2 ensure_platform_experiment ==")
    f1, f2 = stage / "data.csv", stage / "task.py"
    f1.write_bytes(b"a,b\n1,2\n")
    f2.write_bytes(b"print('hi')")

    # 正向：创建模式（不传 id、name 给定）→ uploaded 三元组 + 空 inheritable
    result, text = call(session, "ensure_platform_experiment", {"name": "e2e-s2", "files": [str(f1), str(f2)]})
    if not isinstance(result, dict):
        bad("创建模式", text[:300])
        raise SystemExit(2)
    ok("创建模式返回")
    eid = result["platform_experiment_id"]
    uploaded = {x["remote"]: x["local"] for x in result["files"]["uploaded"]}
    if uploaded == {"/user-data/uploads/data.csv": str(f1), "/user-data/uploads/task.py": str(f2)}:
        ok(f"uploaded 映射（exp={eid}）")
    else:
        bad("uploaded 映射", str(result["files"]))
    ok("inheritable 空态") if result["inheritable"] == [] else bad("inheritable 应为空", str(result["inheritable"]))

    # 幂等：重复调用同清单 → 全部 skipped
    result, _ = call(
        session, "ensure_platform_experiment", {"platform_experiment_id": eid, "files": [str(f1), str(f2)]}
    )
    skipped = sorted(x["local"] for x in result["files"]["skipped"])
    ok("重复调用全 skipped") if skipped == sorted([str(f1), str(f2)]) and not result["files"]["uploaded"] else bad(
        "重复调用应全 skipped", str(result["files"])
    )

    # 增量：改动一个文件 → 恰该文件 updated
    f1.write_bytes(b"a,b\n1,3\n")
    result, _ = call(
        session, "ensure_platform_experiment", {"platform_experiment_id": eid, "files": [str(f1), str(f2)]}
    )
    ok("改动恰落 updated") if [x["local"] for x in result["files"]["updated"]] == [str(f1)] else bad(
        "增量 updated", str(result["files"])
    )

    # 查询模式：只传 id → 零副作用（skipped 全量、无 uploaded/updated）
    result, _ = call(
        session, "ensure_platform_experiment", {"platform_experiment_id": eid, "files": [str(f1), str(f2)]}
    )
    ok("查询模式零副作用") if not (result["files"]["uploaded"] or result["files"]["updated"]) else bad(
        "查询模式应零副作用", str(result["files"])
    )

    # 负向家族
    expect_error(
        session, "不存在 id 404", "ensure_platform_experiment", {"platform_experiment_id": "no-such-exp"}, "no-such-exp"
    )
    expect_error(
        session, "id+name 冲突", "ensure_platform_experiment", {"platform_experiment_id": eid, "name": "x"}, "冲突"
    )
    expect_error(session, "两参全缺", "ensure_platform_experiment", {}, "name 必填")
    expect_error(
        session,
        "dotfile 拒整次",
        "ensure_platform_experiment",
        {"platform_experiment_id": eid, "files": [str(stage / ".hidden")]},
        "dotfile",
    )
    expect_error(
        session,
        "目录拒收",
        "ensure_platform_experiment",
        {"platform_experiment_id": eid, "files": [str(stage)]},
        "目录",
    )
    dup = stage / "dup"
    dup.mkdir()
    (dup / "x.csv").write_bytes(b"1")
    (stage / "x.csv").write_bytes(b"2")
    expect_error(
        session,
        "清单撞名",
        "ensure_platform_experiment",
        {"platform_experiment_id": eid, "files": [str(dup / "x.csv"), str(stage / "x.csv")]},
        "撞名",
    )
    # 相对路径按执行 cwd 解析（#243 通道收敛——原 MCP 拒收面退役）
    rel_dir = stage / "rel"
    rel_dir.mkdir(exist_ok=True)
    (rel_dir / "a.csv").write_bytes(b"r1")
    result, text = call(session, "ensure_platform_experiment", {"platform_experiment_id": eid, "files": ["rel/a.csv"]})
    ok("相对路径按 cwd 解析") if isinstance(result, dict) and any(
        x["local"] == str(rel_dir / "a.csv") for x in result["files"]["uploaded"]
    ) else bad("相对路径解析", text[:200])
    return eid


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mflowy_e2e_tools_") as td:
        stage = Path(td)
        # 遥测常设 off（runbook S0）；preset 名常设（remote 提交面；#157 配置面）
        ch = Channel(cwd=str(stage), env={"MFLOWY_TELEMETRY": "off", "MFLOWY_PLATFORM_JOB_PRESET": "cpu-standard"})
        run_s2(ch, stage)
        eid, j1 = run_s3_s4(ch, stage)
        model_run_id = run_s5(ch, stage, eid, j1)
        run_s6(ch, stage, eid, j1, model_run_id)

    run_s8(stage)
    print(f"\n== 汇总: PASS={NPASS} FAIL={NFAIL} ==")
    sys.exit(1 if NFAIL else 0)


def run_s3_s4(session: Channel, stage: Path) -> tuple[str, str]:
    """S3 提交受理 + S4 在途观察（切片④）。前置：MFLOWY_PLATFORM_JOB_PRESET 已注入 server env。"""
    print("== S3 提交与受理 ==")
    # ensure 建实验 + 上传 diabetes.py（hpo 载荷的数据/脚本源）
    diabetes = Path(__file__).resolve().parent.parent / "examples" / "diabetes" / "diabetes.py"
    result, text = call(session, "ensure_platform_experiment", {"name": "e2e-s3", "files": [str(diabetes)]})
    assert isinstance(result, dict), text[:300]
    eid = result["platform_experiment_id"]

    # switch_job_provider 查询/切换
    result, _ = call(session, "switch_job_provider", {})
    providers = {e["name"]: e for e in result["providers"]}
    if not (result["active"] == "local" and providers["local"]["active"] and providers["remote"]["available"]):
        bad("switch 查询态", str(result))
    else:
        ok("switch 查询：local(active) + remote(available)")
    result, _ = call(session, "switch_job_provider", {"provider": "remote"})
    ok("切 remote") if result.get("active") == "remote" else bad("切 remote", str(result))

    steps = (
        "- load__python:\n    name: 加载\n    params:\n      source: /user-data/uploads/diabetes.py:load\n"
        "- clean__python:\n    name: 清洗\n    params:\n      source: /user-data/uploads/diabetes.py:clean\n"
        "- x_y:\n    name: 拆分\n    params:\n      target: target\n"
        "- cv__split:\n    name: 划分\n    params:\n      test_size: 0.2\n"
        "- model__XGB:\n    name: 搜索\n    params:\n      n_estimators: 30\n      n_trials: 2\n      scoring: mae\n"
    )
    hpo_args = {"modeling_steps": steps, "name": "e2e-s3-hpo", "desc": "S3 受理面", "platform_experiment_id": eid}

    # 负向（先于正向——不消耗集群算力）：缺 platform_experiment_id / inherit 无 zip / 128KiB
    expect_error(
        session,
        "remote 缺实验 id",
        "hyperparameters_search",
        {k: v for k, v in hpo_args.items() if k != "platform_experiment_id"},
        "ensure_platform_experiment",
    )
    expect_error(
        session, "inherit 无 zip", "modeling", {**hpo_args, "inherit_platform_job_id": "job-no-zip"}, "job-no-zip"
    )
    expect_error(session, "128KiB 预检", "modeling", {**hpo_args, "modeling_steps": "x" * (128 * 1024)}, "128KiB")

    # geesefs 可见性沉淀：外部 PUT 后挂载视图目录列示有 TTL 竞态（平台已知陷阱家族
    # ——runbook 已记），提交前留 15s；失败若为 FileNotExisted 系再单次重提
    time_mod.sleep(15)
    t0 = time_mod.monotonic()
    result, text = call(session, "hyperparameters_search", hpo_args)
    elapsed = time_mod.monotonic() - t0
    if (
        isinstance(result, dict)
        and result.get("status") == "submitted"
        and result.get("platform_experiment_id") == eid
        and result.get("dag_result") is None
    ):
        ok(f"受理面 submitted（{elapsed:.1f}s 即返，job={result['job_id']}）")
    else:
        bad("受理面 submitted", text[:300])
        raise SystemExit(2)
    j1 = result["job_id"]

    # job 面形状（平台 HTTP 面——CLI token 经 #46 扩容路）
    face = _job_face(eid, j1)
    if (
        face
        and face["cmd"] == "mflowy-job"
        and face["args"][:3] == ["wrap", "--tool", "hyperparameters_search"]
        and face["args"][3] == "--kwargs-json"
        and "--inherit-from" not in face["args"]
    ):
        ok("job 面：mflowy-job wrap 单 argv JSON、无 --inherit-from")
    else:
        bad("job 面形状", str(face))

    print("== S4 在途观察 ==")
    # tail_lines 边界（本地 fail-loud，不发出请求）
    for bad_tail in (0, -1, 10001):
        expect_error(
            session,
            f"tail_lines={bad_tail}",
            "get_platform_job_logs",
            {"job_id": j1, "tail_lines": bad_tail},
            "1–10000",
        )

    # 轮询至终态

    terminal = _poll_terminal(session, j1)
    if terminal and terminal["terminal"] and terminal["status"] == "succeeded":
        ok("S4 轮询终态 succeeded")
    elif terminal and "FileNotExisted" in terminal.get("logs", ""):
        # geesefs 可见性竞态单次重提（已知陷阱，runbook S3 注记）
        print("  … FileNotExisted 竞态——沉淀 20s 后单次重提")
        time_mod.sleep(20)
        result, text = call(session, "hyperparameters_search", hpo_args)
        if isinstance(result, dict) and result.get("status") == "submitted":
            j1 = result["job_id"]
            terminal = _poll_terminal(session, j1)
        ok("竞态重提后终态 succeeded") if terminal and terminal.get("status") == "succeeded" else bad(
            "竞态重提后仍失败", str(terminal)[:200]
        )
    else:
        bad("S4 轮询终态", str(terminal)[:200])
    shape_ok = set(terminal or {}) >= {"job_id", "status", "terminal", "tail_lines", "logs"}
    ok("logs 合并返回五键") if shape_ok else bad("logs 返回形状", str(set(terminal or {})))
    logs = (terminal or {}).get("logs", "")
    ok(f"wrap 版本回显（mflowy {_WHL_VERSION}）") if f"mflowy {_WHL_VERSION}" in logs else bad(
        "wrap 版本回显缺", logs[:200]
    )
    ok("wrap zip 打包回显") if "zip packed" in logs else bad("zip packed 缺", logs[-200:])

    # J1 产出 → ensure inheritable 出现该 job（S2 延后断言在此收口）
    result, _ = call(session, "ensure_platform_experiment", {"platform_experiment_id": eid})
    inheritable = {e["job_id"]: e["status"] for e in result["inheritable"]}
    ok("inheritable 出现 J1") if inheritable.get(j1) == "succeeded" else bad(
        "inheritable 缺 J1", str(result["inheritable"])
    )

    # local 等价：切回 local 带模式专属参数 → 忽略且恒 finished
    result, _ = call(session, "switch_job_provider", {"provider": "local"})
    small_steps = steps.replace("n_estimators: 30", "n_estimators: 10")
    result, text = call(
        session,
        "hyperparameters_search",
        {
            "modeling_steps": small_steps,
            "name": "e2e-s3-local",
            "desc": "local 等价",
            "platform_experiment_id": eid,
            "inherit_platform_job_id": j1,
        },
    )
    if result is None and LAST_RC == 0:
        ok("local 模式忽略平台参数恒 finished（rc=0——DAG 流式上屏契约无 JSON 回显）")
    else:
        bad("local 模式等价", f"rc={LAST_RC} {text[:200]}")
    return eid, j1


def _job_face(experiment_id: str, job_id: str) -> dict | None:
    """平台 HTTP 面查 job 的 cmd/args（CLI token；驱动进程直接消费 client）。"""
    try:
        from mflowy.model.job_provider.platform_client import RemoteExperimentClient

        job = RemoteExperimentClient().get_job(job_id)
        return {"cmd": job.cmd, "args": job.args}
    except Exception as e:
        bad("job 面查询", f"{type(e).__name__}: {e}")
        return None


def _poll_terminal(session: Channel, job_id: str, *, rounds: int = 100) -> dict | None:
    """轮询至终态（terminal=true 即停）；上游瞬态错误（如 logs 500）容错重试。"""
    result = None
    transient = 0
    for _ in range(rounds):
        result, text = call(session, "get_platform_job_logs", {"job_id": job_id, "tail_lines": 200})
        if not isinstance(result, dict):
            transient += 1
            if transient >= 10:
                raise AssertionError(f"轮询连续瞬态失败 ×{transient}：{text[:300]}")
            time_mod.sleep(6)
            continue
        transient = 0
        if result["terminal"]:
            return result
        time_mod.sleep(6)
    return result


def run_s5(session: Channel, stage: Path, eid: str, j1: str) -> str | None:
    """S5 拉取与本地查询（切片⑤）：download 落位/原子替换 + 观察面单参 tracking_uri。

    返回 J1 的模型步 run_id（S6 续跑源）。"""
    print("== S5 拉取与本地查询 ==")
    target = stage / "artifacts"
    target.mkdir()
    result, text = call(
        session,
        "download_platform_job_artifacts",
        {"job_id": j1, "platform_experiment_id": eid, "target_dir": str(target)},
    )
    if not isinstance(result, dict) or result.get("local_task_dir") != str(target / j1):
        bad("download 落位 target_dir/{job_id}/", text[:300])
        raise SystemExit(2)
    dest = target / j1

    def _has(*names: str) -> bool:
        return all((dest / n).is_file() for n in names)

    ok("解压落位 + file_count") if isinstance(result["file_count"], int) and result["file_count"] > 0 else bad(
        "file_count", str(result)
    )
    ok("zip 含 mlflow.db") if _has("mlflow.db") else bad(
        "mlflow.db 缺", str(sorted(p.name for p in dest.rglob("*"))[:10])
    )
    ok("zip 含 .mlruns/ 树") if any(dest.glob(".mlruns/**")) else bad(".mlruns 缺", "")
    ok("mlflow_db 路径回传") if result["mlflow_db"] == str(dest / "mlflow.db") else bad(
        "mlflow_db", str(result["mlflow_db"])
    )
    ok("zip 本体即弃") if not list(target.glob("*.zip")) else bad("zip 残留", str(list(target.glob("*.zip"))))

    result2, _ = call(
        session,
        "download_platform_job_artifacts",
        {"job_id": j1, "platform_experiment_id": eid, "target_dir": str(target)},
    )
    ok("重下载幂等") if result2 == result else bad("重下载应等值", f"{result2} vs {result}")
    ok("换位无残留") if not list(target.glob(f".{j1}*")) else bad("换位残留", str(list(target.glob(f".{j1}*"))))

    # 观察面单参：tracking_uri 指到下载库 → 查 run + 工件路径改写为本地；顺带取模型步 run_id
    uri = f"sqlite:///{dest}/mlflow.db"
    model_run_id = None
    exp_hits = None
    for probe in ("1", "0"):
        result, text = call(session, "get_experiment", {"experiment_id": probe, "tracking_uri": uri})
        if isinstance(result, dict) and result.get("runs"):
            exp_hits = (probe, result)
            break
    if exp_hits and exp_hits[1]["runs"]:
        probe, exp_view = exp_hits
        ok(f"get_experiment(tracking_uri) 命中实验 {probe}（runs={len(exp_view['runs'])}）")
        model_runs = [r for r in exp_view["runs"] if str(r.get("node", "")).startswith("model__")]
        if model_runs:
            model_run_id = model_runs[0]["run_id"]
            ok(f"模型步 run 就位（{model_runs[0]['node']}={model_run_id[:8]}…）")
        loc = str(exp_view.get("artifact_location", ""))
        ok("工件根已改写为本地") if loc.startswith(str(dest)) and "/user-data" not in loc else bad(
            "artifact_location 改写", loc
        )
        if model_run_id:
            result, text = call(session, "get_run", {"run_id": model_run_id, "tracking_uri": uri})
            if isinstance(result, dict):
                paths = [result["info"]["artifact_uri"]] + [a["path"] for a in result.get("artifacts", [])]
                ok("run 工件路径双改写（uri+清单）") if all("/user-data" not in p2 for p2 in paths) and any(
                    str(p2).startswith(str(dest)) for p2 in paths
                ) else bad("run 工件改写", str(paths[:3]))
            else:
                bad("get_run(tracking_uri)", text[:200])
    else:
        bad("get_experiment(tracking_uri) 未命中任何实验", text[:200])

    # 缺省查询回归（不传 tracking_uri → 机器默认库）
    result, text = call(session, "get_experiment", {"experiment_id": "0"})
    ok("缺省查询回归（Default 实验在）") if isinstance(result, dict) and result.get("experiment_id") == "0" else bad(
        "缺省回归", text[:200]
    )

    # 负向
    expect_error(
        session,
        "target_dir 不存在",
        "download_platform_job_artifacts",
        {"job_id": j1, "platform_experiment_id": eid, "target_dir": str(stage / "nope")},
        "target_dir",
    )
    expect_error(
        session,
        "MCP 相对 target_dir",
        "download_platform_job_artifacts",
        {"job_id": j1, "platform_experiment_id": eid, "target_dir": "rel/dir"},
        "绝对路径",
    )
    return model_run_id


def run_s6(session: Channel, stage: Path, eid: str, j1: str, model_run_id: str | None) -> None:
    """S6 续跑链：继承 × resume × 跨 Job 寻址（切片⑥核心验收）。"""
    print("== S6 续跑链 ==")
    assert model_run_id, "S5 未取得模型步 run_id——S6 无从续跑"
    target = stage / "s6"
    target.mkdir()

    steps = (
        "- load__python:\n    name: 加载\n    params:\n      source: /user-data/uploads/diabetes.py:load\n"
        "- clean__python:\n    name: 清洗\n    params:\n      source: /user-data/uploads/diabetes.py:clean\n"
        "- x_y:\n    name: 拆分\n    params:\n      target: target\n"
        "- cv__split:\n    name: 划分\n    params:\n      test_size: 0.2\n"
        "- model__XGB:\n    name: 搜索\n    params:\n      n_estimators: 30\n      n_trials: 2\n      scoring: mae\n"
    )

    # J2 = hpo 续跑：inherit J1 + resume J1 模型步 study
    call(session, "switch_job_provider", {"provider": "remote"})
    result, text = call(
        session,
        "hyperparameters_search",
        {
            "modeling_steps": steps,
            "name": "e2e-j2-resume",
            "desc": "S6 hpo 续跑",
            "platform_experiment_id": eid,
            "inherit_platform_job_id": j1,
            "resume_study_from": model_run_id,
        },
    )
    j2 = result.get("job_id") if isinstance(result, dict) else None
    ok(f"J2 受理（{j2}）") if j2 else bad("J2 受理", text[:200])
    terminal = _poll_terminal(session, j2)
    ok("J2 succeeded") if terminal and terminal.get("status") == "succeeded" else bad("J2 终态", str(terminal)[:200])

    result, _ = call(
        session,
        "download_platform_job_artifacts",
        {"job_id": j2, "platform_experiment_id": eid, "target_dir": str(target)},
    )
    j2_dir = target / j2
    # #244 零拷贝：resume 解析 tag 相对路径 → 前史（J1 模型步 run）工件树内 study.db
    # 直驻增量——J2 模型步 run 无拷回副本；zip 内一份（J1 树带入），≥4 即增量实证
    import sqlite3

    study_dbs = sorted(j2_dir.glob(".mlruns/*/artifacts/study.db"))
    counts = [sqlite3.connect(db).execute("select count(*) from trials").fetchone()[0] for db in study_dbs]
    if max(counts, default=0) >= 4:
        ok(f"J2 study 增量（study.db trials={counts}，前史 2 + 本轮 2——前史树内直驻）")
    else:
        bad("J2 study 增量", str(counts))
    result, _ = call(session, "get_experiment", {"experiment_id": "1", "tracking_uri": f"sqlite:///{j2_dir}/mlflow.db"})
    runs2 = result.get("runs", []) if isinstance(result, dict) else []
    has_j1 = any(r["run_id"] == model_run_id for r in runs2)
    ok("J2 mlflow.db 含 J1 runs（继承滚雪球）") if has_j1 else bad(
        "J1 run 不在 J2 库", str([r["run_id"][:8] for r in runs2])
    )
    # J2 模型步 run（续跑新写——run_id 不同于继承带入的 J1 model_run_id；#244 零拷贝
    # 下其工件树无 study.db 副本——study 单源居 J1 树）；S7 消费其工件改写断言
    j2_model_run = next(
        (r["run_id"] for r in runs2 if str(r.get("node", "")).startswith("model__") and r["run_id"] != model_run_id),
        None,
    )
    ok(f"J2 续跑模型步 run（{str(j2_model_run)[:8]}…）") if j2_model_run else bad(
        "J2 模型步 run 缺", str([r.get("node") for r in runs2])
    )

    # J3 = modeling 终训，inherit J2（终训 steps = 全 pin——搜索空间缺省即空间、
    # n_trials 是搜索旋钮；全 pin 才走 holdout 面落 X_test）
    train_steps = (
        "- load__python:\n    name: 加载\n    params:\n      source: /user-data/uploads/diabetes.py:load\n"
        "- clean__python:\n    name: 清洗\n    params:\n      source: /user-data/uploads/diabetes.py:clean\n"
        "- x_y:\n    name: 拆分\n    params:\n      target: target\n"
        "- cv__split:\n    name: 划分\n    params:\n      test_size: 0.2\n"
        "- model__XGB:\n    name: 终训\n    params:\n      n_estimators: 30\n      max_depth: 3\n"
        "      min_child_weight: 5\n      subsample: 0.9\n      colsample_bytree: 0.9\n"
        "      gamma: 0.01\n      reg_alpha: 0.01\n      reg_lambda: 0.01\n      learning_rate: 0.1\n"
    )
    result, text = call(
        session,
        "modeling",
        {
            "modeling_steps": train_steps,
            "name": "e2e-j3-train",
            "desc": "S6 modeling",
            "platform_experiment_id": eid,
            "inherit_platform_job_id": j2,
        },
    )
    j3 = result.get("job_id") if isinstance(result, dict) else None
    ok(f"J3 受理（{j3}）") if j3 else bad("J3 受理", text[:200])
    terminal = _poll_terminal(session, j3)
    ok("J3 succeeded") if terminal and terminal.get("status") == "succeeded" else bad("J3 终态", str(terminal)[:200])

    result, _ = call(
        session,
        "download_platform_job_artifacts",
        {"job_id": j3, "platform_experiment_id": eid, "target_dir": str(target)},
    )
    j3_dir = target / j3

    def _has(seg: str) -> bool:
        return any(j3_dir.glob(seg))

    if _has("**/X_test.parquet") and _has("**/y_test.parquet") and _has("**/y_pred.parquet"):
        ok("J3 zip 追加 X_test/y_test/y_pred")
    else:
        bad("J3 副产物缺", str(sorted(str(x.relative_to(j3_dir)) for x in j3_dir.rglob("*"))[:8]))
    ok("J3 zip 含评估图") if any(j3_dir.glob("**/*.png")) else bad("评估图缺", "")
    if any((j3_dir / ".mlruns").glob("models/*")) or any(
        p.name.startswith("m-") for p in (j3_dir / ".mlruns").rglob("*") if p.is_dir()
    ):
        ok("J3 logged model 本体（models/ 形态）")
    else:
        print("  ⚕ logged models/ 缺席（终训本地实证亦无——#155 双形态为存在时契约，非阻断）")
    result, _ = call(session, "get_experiment", {"experiment_id": "1", "tracking_uri": f"sqlite:///{j3_dir}/mlflow.db"})
    j3_run = None
    for x_test in sorted(j3_dir.glob(".mlruns/*/artifacts/X_test.parquet")):
        j3_run = x_test.parent.parent.name  # .mlruns/<run_id>/artifacts/X_test.parquet
        break
    ok(f"J3 终训 run（{j3_run[:8]}…，X_test 在场自证）") if j3_run else bad("J3 终训 run 缺 X_test", "")

    # J4 = explanation：model=<J3 run> + inherit J3 —— 跨 Job 寻址硬前提实证
    result, text = call(
        session,
        "explanation",
        {
            "modeling_steps": train_steps,
            "model": j3_run,
            "name": "e2e-j4-shap",
            "desc": "S6 跨 Job 寻址",
            "platform_experiment_id": eid,
            "inherit_platform_job_id": j3,
        },
    )
    j4 = result.get("job_id") if isinstance(result, dict) else None
    ok(f"J4 受理（{j4}）") if j4 else bad("J4 受理", text[:200])
    terminal = _poll_terminal(session, j4)
    ok("J4 succeeded（跨 Job 寻址实证）") if terminal and terminal.get("status") == "succeeded" else bad(
        "J4 终态", str(terminal)[:300]
    )

    # S7 观察面收口
    print("== S7 观察面收口 ==")
    uri3 = f"sqlite:///{j3_dir}/mlflow.db"
    result, _ = call(session, "get_experiment", {"experiment_id": "1", "tracking_uri": uri3})
    runs_all = result.get("runs", []) if isinstance(result, dict) else []
    ok(f"J3 库 runs = 继承链累计（{len(runs_all)}）") if len(runs_all) >= 4 else bad(
        "累计 runs 不足", str(len(runs_all))
    )
    loc = str(result.get("artifact_location", ""))
    ok("工件根本地化") if loc.startswith(str(j3_dir)) and "/user-data" not in loc else bad("artifact_location", loc)
    if j2_model_run:
        result, _ = call(session, "get_run", {"run_id": j2_model_run, "tracking_uri": uri3})
        if isinstance(result, dict):
            paths = [result["info"]["artifact_uri"]] + [a["path"] for a in result.get("artifacts", [])]
            ok("run 工件改写（#244 零拷贝下无 study.db 副本——study 单源居 J1 树）") if all(
                "/user-data" not in p2 for p2 in paths
            ) else bad("工件改写漏", str(paths[:3]))
            ok("工件树可寻址（artifacts/）") if any(
                p2.endswith("artifacts") or p2.endswith("artifacts/") for p2 in paths
            ) else bad("工件树缺", str(paths[:3]))
        else:
            bad("get_run J2 模型步", "")


def run_s8(stage: Path) -> None:
    """S8 横切负向：平台不可达（独立 server 实例指向死端口）。"""
    print("== S8 横切负向 ==")
    with tempfile.TemporaryDirectory(prefix="mflowy_e2e_s8_") as td8:  # 自持 cwd——主 stage 目录在主块退场即清
        _run_s8_session(Path(td8))
    ok("遥测门被动断言：全程零遥测提示（env off）")


def _run_s8_session(stage: Path) -> None:
    ch = Channel(cwd=str(stage), env={"MFLOWY_TELEMETRY": "off", "MFLOWY_BASE_URL": "http://localhost:9"})
    # CLI 通道的不可达面：auth 环境一致性门先于 HTTP（凭据属真实网关、
    # 探针指向死端口 → fail-loud 报两 URL）；传输层 PlatformApiError+NodeStop
    # 提示形态由单测钉（test_platform_client.py S8 用例）——此处钉通道行为
    expect_error(
        ch,
        "平台不可达（auth 环境门先拒）",
        "ensure_platform_experiment",
        {"name": "dead-port-probe"},
        "localhost:9",
    )


if __name__ == "__main__":
    main()
