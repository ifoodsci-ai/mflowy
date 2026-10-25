"""e2e-bench 三基准冒烟门（#75 决议，发布前人肉门——不进 CI）。

链面（按任务裁剪）：
- iris    ：全五族 DAG 搜索行为（examples/iris/iris.yaml）
- digits  ：scaler → MLP training 行为 + XGB 搜索（examples/digits/digits.yaml）
- titanic ：全清洗链 → 三树族搜索（examples/titanic/titanic.yaml）
- 会话链  ：iris/RF 分类会话 start → select_model → run_hpo → final_train_test（B2 执行位）

判据 = 节点全绿 + 每任务指标下限（防「绿但退化」；下限 = benchmark_readiness_report.md
实测折，见 THRESHOLDS 注释）。越限/未完成 → FAIL 行 + 非零退出。
数据全部来自 examples/<task>/<task>.py 生成器（#74 决议），无网络、无仓外依赖；
追踪默认落临时 file store（零仓库产物，M10），已设 MLFLOW_TRACKING_URI 则尊重。
需 ``--extra deep``（digits 链含 MLP）。

越限自证（验收标准 2）：``BENCH_FLOOR_BUMP='iris|RF|99' uv run --extra model --extra deep python scripts/_e2e_bench.py``
——人为抬下限应得 FAIL 行与非零退出。
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# 相对 source 绝对化单源（scripts 目录同居件互引；E402 不可避免——需先改 sys.path）
from _e2e_payload import absolutize_load_sources  # noqa: E402

# 指标下限（u_inner，f1——higher-better）：报告实测 iris RF=0.958 / digits XGB=0.951 /
# titanic 三树族≈0.64，取保守折。BENCH_FLOOR_BUMP（task|family|floor）供越限自证。
THRESHOLDS: dict[str, dict[str, float]] = {
    "iris": {"RF": 0.90},
    "digits": {"XGB": 0.90},
    "titanic": {"XGB": 0.60, "LGBM": 0.60, "CAT": 0.60},
}

failures: list[str] = []


def _report(ok: bool, line: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {line}", flush=True)
    if not ok:
        failures.append(line)


def _bumped_floor(task: str, family: str, floor: float) -> float:
    for part in os.environ.get("BENCH_FLOOR_BUMP", "").split():
        t, f, v = ([*part.split("|"), "", ""])[:3]
        if t == task and f == family:
            return float(v)
    return floor


def _check_evidence(task: str, family: str, evidence, floor: float) -> None:
    floor = _bumped_floor(task, family, floor)
    u = evidence.get("u_inner") if isinstance(evidence, dict) else None
    _report(u is not None and u >= floor, f"{task}/{family} u_inner={u} floor={floor}（搜索行为证据）")


def _iris_df():
    spec = importlib.util.spec_from_file_location("e2e_bench_iris", ROOT / "examples" / "iris" / "iris.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load()


def dag_chains(provider, tmp: str) -> None:
    for task, floors in THRESHOLDS.items():
        # 生成器复制出仓（tmp）再绝对引用——工件锚 tmp/.mlruns，零仓库产物（M10）
        staging = pathlib.Path(tmp) / "tasks" / task
        staging.mkdir(parents=True, exist_ok=True)
        generator_src = ROOT / "examples" / task / f"{task}.py"
        generator = staging / f"{task}.py"
        generator.write_bytes(generator_src.read_bytes())
        yaml_path = ROOT / "examples" / f"{task}/{task}.yaml"
        steps = absolutize_load_sources(yaml_path.read_text(encoding="utf-8"), yaml_path.parent).replace(
            str(yaml_path.parent), str(staging)
        )
        result = asyncio.run(
            provider.modeling(modeling_steps=steps, name=f"e2e-bench:{task}", desc=f"基准冒烟门 {task} 链")
        )
        if result.status != "finished":
            _report(False, f"{task} DAG status={result.status} error={getattr(result, 'error', None)}")
            continue
        finished = [r for r in result.runs if r.status == "FINISHED"]
        _report(len(finished) == len(result.runs), f"{task} 节点全绿（{len(finished)}/{len(result.runs)}）")
        for family, floor in floors.items():
            hit = next((r for r in finished if r.node == f"model__{family}"), None)
            if hit is None:
                _report(False, f"{task}/{family} 无 FINISHED run（候选 {[r.node for r in result.runs]}）")
                continue
            _check_evidence(task, family, hit.value, floor)


def session_chain(tmp: str) -> None:
    """B2 执行位：分类会话最小链（run_hpo → final_train_test）。会话根强制临时（一次性）。"""
    csv = os.path.join(tmp, "iris.csv")
    _iris_df().to_csv(csv, index=False)
    sessions_root = os.path.join(tmp, "sessions")
    os.environ["MFLOWY_MODELING_SESSIONS"] = sessions_root
    from mflowy.model import session as ms

    s = ms.start_session(
        csv,
        "cv: 3",
        card={"domain": "e2e-bench", "target": "target"},
        task="classification",
        metric="f1",
        max_steps=8,
        candidates=[{"family": "RF", "baseline": True}],
    )
    ms.apply_action(s["session_id"], "select_model", {"family": "RF"})
    ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 2})
    ms.apply_action(s["session_id"], "final_train_test")
    manifest = json.loads((pathlib.Path(sessions_root) / s["session_id"] / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["models"]["RF|passthrough|passthrough"]
    u_inner, u_final = entry.get("u_inner"), entry.get("u")
    _report(
        u_inner is not None and u_final is not None,
        f"session/RF 分类链 u_inner={u_inner} holdout_u={u_final}（B2 执行位）",
    )


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="mflowy_bench_")
    # sqlite 隔离库（file store 已进 mlflow 维护模式；默认 ~/.mflowy/mlflow.db 是机器级共享
    # 路径，任何其他环境写入都会毒化——P2 教训）。已设 MLFLOW_TRACKING_URI 则尊重
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{tmp}/bench.db")
    from mflowy.model.job_provider import get_job_provider

    dag_chains(get_job_provider(), tmp)
    session_chain(tmp)
    print(f"\ne2e-bench: {'FAIL ' + str(len(failures)) + ' 项' if failures else '全绿'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
