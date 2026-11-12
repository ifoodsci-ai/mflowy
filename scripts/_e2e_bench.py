"""e2e-bench 六基准冒烟门（#75 决议，发布前人肉门——不进 CI；工业三链随 #74 范式扩展）。

链面（按任务裁剪）：
- iris    ：全五族 DAG 搜索行为（examples/iris/iris.yaml）
- digits  ：scaler → MLP training 行为 + XGB 搜索（examples/digits/digits.yaml）
- titanic ：全清洗链 → 三树族搜索（examples/titanic/titanic.yaml）
- bearing_rul ：机组退化 RUL 回归（RF/XGB，mae——回归判据 = u_inner 已对齐
  higher-better，floor 为负 MAE，越不负 = MAE 越小）
- steel_defect：类不平衡质检分类（onehot → 三树族，f1）
- soft_sensor ：传感器缺失脏表清洗（fill_median）→ onehot → XGB/RF 回归（mae）
- 会话链  ：iris/RF 分类会话 start → select_model → run_hpo → final_train_test（B2 执行位）
- 应用面  ：终训 run_id 寻址跑 explanation/predict/inverse_optimization（#167——判据 =
  三工具可调 + shap_values/prediction/search_results parquet 落 run 工件树，无指标下限）

判据 = 节点全绿 + 每任务指标下限（防「绿但退化」；下限 = benchmark_readiness_report.md
实测折 + 工业三链实测折，见 THRESHOLDS 注释）。越限/未完成 → FAIL 行 + 非零退出。
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

# 指标下限（u_inner——higher-better 对齐；分类 = f1，回归 = 负 MAE）：报告实测
# iris RF=0.958 / digits XGB=0.951 / titanic 三树族≈0.64；工业三链实测（数据种子 +
# machinery seed 双确定性，逐次重跑数值逐位一致）：bearing RF=-10.60 / XGB=-7.07
# （MAE，循环数）、steel 三树族≈0.886-0.890（f1）、soft XGB=-0.352 / RF=-0.566
# （MAE，纯度百分点）。均取保守折（结构塌方如任务误判/特征失效必越限——退化到
# 均值预测时 bearing MAE≈150+）。BENCH_FLOOR_BUMP（task|family|floor）供越限自证。
THRESHOLDS: dict[str, dict[str, float]] = {
    "iris": {"RF": 0.90},
    "digits": {"XGB": 0.90},
    "titanic": {"XGB": 0.60, "LGBM": 0.60, "CAT": 0.60},
    "bearing_rul": {"RF": -20.0, "XGB": -15.0},  # 实测 -10.60 / -7.07
    "steel_defect": {"XGB": 0.75, "LGBM": 0.75, "CAT": 0.75},  # 实测 ≈0.89
    "soft_sensor": {"XGB": -0.60, "RF": -0.80},  # 实测 -0.352 / -0.566
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
        if result.status != "finished" or result.dag_result is None:
            _report(False, f"{task} DAG status={result.status} error={getattr(result, 'error', None)}")
            continue
        runs = result.dag_result.runs  # #204 起 modeling 受理面 = JobOutcome（DAG 目录内嵌 finished）
        finished = [r for r in runs if r.status == "FINISHED"]
        _report(len(finished) == len(runs), f"{task} 节点全绿（{len(finished)}/{len(runs)}）")
        for family, floor in floors.items():
            hit = next((r for r in finished if r.node == f"model__{family}"), None)
            if hit is None:
                _report(False, f"{task}/{family} 无 FINISHED run（候选 {[r.node for r in runs]}）")
                continue
            _check_evidence(task, family, hit.value, floor)


def session_chain(tmp: str) -> tuple[str, str]:
    """B2 执行位：分类会话最小链（run_hpo → final_train_test）。会话根强制临时（一次性；
    #207 锚定链——锚 tmp，根 = tmp/modeling_sessions，一次性脚本不还锚）。"""
    csv = os.path.join(tmp, "iris.csv")
    _iris_df().to_csv(csv, index=False)
    from mflowy.model import session as ms
    from mflowy.utils.path import pin_task_dir

    pin_task_dir(tmp)
    sessions_root = os.path.join(tmp, "modeling_sessions")

    s = ms.start_session(
        csv,
        "cv: 3",
        card={"domain": "e2e-bench", "target": "target"},
        task="classification",
        metric="f1",
        max_steps=8,
        candidates=[{"family": "RF", "baseline": True}],
    )
    sid = s.session_id  # #89 会话基座：start/apply 响应 = Observation 封闭模型（属性面，非 dict）
    ms.apply_action(sid, "select_model", {"family": "RF"})
    ms.apply_action(sid, "run_hpo", {"n_trials": 2})
    ms.apply_action(sid, "final_train_test")
    # 账本板面读 board.json（域状态）——manifest.json 随 #89 溶解为 observation.json + board.json
    board = json.loads((pathlib.Path(sessions_root) / sid / "board.json").read_text(encoding="utf-8"))
    entry = board["models"]["RF|passthrough|passthrough"]
    u_inner, u_final = entry.get("u_inner"), entry.get("u")
    _report(
        u_inner is not None and u_final is not None,
        f"session/RF 分类链 u_inner={u_inner} holdout_u={u_final}（B2 执行位）",
    )
    return sid, sessions_root


def application_face(provider, tmp: str, sid: str, sessions_root: str) -> None:
    """#81/#167 应用面三工具冒烟：以会话终训 run_id 寻址（跨 run 消费恰是应用面形态）。

    判据 = 三工具可调 + 工件落 run artifact 树（shap_values / prediction /
    search_results parquet）；不设指标下限（发布门保可调性，语义面归 make e2e 人判）。
    """
    csv = os.path.join(tmp, "iris.csv")
    board = json.loads((pathlib.Path(sessions_root) / sid / "board.json").read_text(encoding="utf-8"))
    entry = board["models"]["RF|passthrough|passthrough"]
    run_id = entry["run_id"]  # 终训 run（name='final' 模型 + 副产物——应用面寻址源）
    plan_text = sorted((pathlib.Path(sessions_root) / sid / "plans").glob("final_*.yaml"))[-1].read_text(
        encoding="utf-8"
    )

    def _art_hit(result, filename: str):
        if result.status != "finished" or result.dag_result is None:
            return None
        return next((r for r in result.dag_result.runs if filename in r.artifacts), None)

    r1 = asyncio.run(
        provider.explanation(
            modeling_steps=plan_text, model=run_id, name="e2e-bench:explanation", desc="应用面冒烟 SHAP"
        )
    )
    _report(
        _art_hit(r1, "shap_values.parquet") is not None,
        f"application/explanation shap_values.parquet 落 run（终训 {run_id[:8]} 寻址，status={r1.status}）",
    )
    r2 = asyncio.run(provider.predict(data=csv, model=run_id))
    _report(
        _art_hit(r2, "prediction.parquet") is not None,
        f"application/predict prediction.parquet 落 run（model={run_id[:8]}）",
    )
    # n_trials=50 控时长（#77 后 n_trials 真实生效——冒烟不需深搜）
    r3 = asyncio.run(
        provider.inverse_optimization(data=csv, model=run_id, direction={"target": "maximize"}, n_trials=50)
    )
    _report(
        _art_hit(r3, "search_results.parquet") is not None,
        "application/inverse_optimization search_results.parquet 落 run（n_trials=50）",
    )


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="mflowy_bench_")
    # sqlite 隔离库（file store 已进 mlflow 维护模式；默认 ~/.mflowy/mlflow.db 是机器级共享
    # 路径，任何其他环境写入都会毒化——P2 教训）。已设 MLFLOW_TRACKING_URI 则尊重
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{tmp}/bench.db")
    from mflowy.model.job_provider import get_job_provider

    provider = get_job_provider()
    dag_chains(provider, tmp)
    sid, sessions_root = session_chain(tmp)
    application_face(provider, tmp, sid, sessions_root)
    print(f"\ne2e-bench: {'FAIL ' + str(len(failures)) + ' 项' if failures else '全绿'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
