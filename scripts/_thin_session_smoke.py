"""瘦装门会话远程冒烟（#218 第 5 条，CI thin job 消费）：mock provider 受理下
modeling_* 会话工具全链可用——start → select_model → run_hpo → observe。

- donor run 预置：真跑 RF search（sklearn-only 族，瘦装可跑；study live 窗口 +
  mflowy.node tag）——mock provider 以 finished JobOutcome 指向它
- CLI 子进程四连（真实旗标编译面）；断言账本 u_inner / observation_json 往返
- 全程零平台（mock 零 submitted）、零本地重栈（RF 族 sklearn 面）

用法（CI thin job / 本地瘦装环境）::

    MFLOWY_JOB_PROVIDER=session_mock:SessionMockProvider \
    THIN_DONOR_RUN=<donor run_id> PYTHONPATH=<mock 目录> \
    uv run [--project <repo>] python scripts/_thin_session_smoke.py
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


def _cli(*args: str) -> dict:
    """mflowy CLI 子进程调用：stdout 单 JSON 文档（查询类回显——业务数据通道）。"""
    proc = subprocess.run(["mflowy", *args], capture_output=True, text=True, env={**os.environ}, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"CLI 失败（{args[0]}）：rc={proc.returncode}\nstderr:\n{proc.stderr[-2000:]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise SystemExit(f"CLI stdout 非单 JSON 文档（{args[0]}）：{e}\nstdout 尾：\n{proc.stdout[-500:]}") from e


def build_donor(csv_path: str) -> str:
    """donor 模型步 run：真跑 RF search（study live 窗口 = provider 形态）。"""
    import mlflow
    import numpy as np
    import pandas as pd
    from mflowy.data.types import TASKTYPE
    from mflowy.model.cv.split import split as split_fn
    from mflowy.model.model._split import materialize_split
    from mflowy.model.model.random_forest import RF
    from mflowy.model.model.study import reset_study_live, set_study_live
    from mflowy.model.model.types import MetricName
    from mflowy.model.x_y import XySpec

    rng = np.random.default_rng(4)
    x = rng.normal(size=60)
    df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=60)})
    xy = XySpec(df[["x"]], df[["target"]], TASKTYPE.REGRESSION)
    ts = materialize_split(split_fn(x_y=xy, cv=3), xy)
    token = set_study_live(True)
    try:
        with mlflow.start_run(run_name="donor:search") as run:
            RF(ts, n_trials=2, scoring=MetricName.MAE)
            mlflow.set_tag("mflowy.node", "model__RF")
    finally:
        reset_study_live(token)
    return run.info.run_id


_MOCK_MODULE = '''"""瘦装门会话 mock provider：finished 受理（donor run 桩——零平台零本地重栈依赖）。"""

import os


class SessionMockProvider:
    async def hyperparameters_search(self, *, modeling_steps, name, desc, **kw):
        from mflowy.driver.context import NodeResult
        from mflowy.driver.dag import DAGResult
        from mflowy.model.job_provider.protocol import JobOutcome

        return JobOutcome.finished(
            dag_result=DAGResult(
                experiment_id="0",
                status="finished",
                error="",
                graph="",
                runs=[
                    NodeResult(
                        node="model__RF", name="会话搜索-RF",
                        run_id=os.environ["THIN_DONOR_RUN"], status="FINISHED", artifacts=[], output="",
                    )
                ],
            )
        )
'''


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="thin_session_") as td:
        csv = Path(td) / "reg.csv"
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(4)
        x = rng.normal(size=60)
        pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=60)}).to_csv(csv, index=False)

        mock_dir = Path(td) / "mock"
        mock_dir.mkdir()
        (mock_dir / "session_mock.py").write_text(_MOCK_MODULE, encoding="utf-8")
        # 环境自隔离：tracking 与会话根都落 td（不污染用户库 / ~/.mflowy——本地复跑零残留）
        os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{Path(td) / 'mlflow.db'}")
        os.environ.setdefault("MFLOWY_TASK_DIR", td)

        donor = build_donor(str(csv))
        os.environ.update(
            {
                "MFLOWY_JOB_PROVIDER": "session_mock:SessionMockProvider",
                "THIN_DONOR_RUN": donor,
                "PYTHONPATH": str(mock_dir),
            }
        )

        # start → observation_json 往返
        started = _cli(
            "modeling_load_or_start",
            "--kwargs-json",
            json.dumps(
                {
                    "file_path": str(csv),
                    "cv_card": "cv: 3",
                    "target": "target",
                    "task": "regression",
                    "metric": "mae",
                    "candidates": [{"family": "RF", "baseline": True}],
                },
                ensure_ascii=False,
            ),
        )
        observation_json = started["observation_json"]
        assert observation_json and Path(observation_json).is_file(), started
        anchor = json.dumps({"observation_json": observation_json}, ensure_ascii=False)

        # select_model（配置动作零步费）→ run_hpo（mock finished 受理）→ observe
        _cli(
            "modeling_apply",
            "--kwargs-json",
            json.dumps(
                {"observation_json": observation_json, "node": "select_model", "params": {"family": "RF"}},
                ensure_ascii=False,
            ),
        )
        r = _cli(
            "modeling_apply",
            "--kwargs-json",
            json.dumps(
                {
                    "observation_json": observation_json,
                    "node": "run_hpo",
                    "params": {"n_trials": 2, "platform_experiment_id": "exp-thin"},
                },
                ensure_ascii=False,
            ),
        )
        models = r["biz"]["board"]
        assert any(e.get("u_inner") is not None for e in models.values()), f"run_hpo 后账本无 u_inner：{models}"
        assert r["validation"]["steps"] == 1, r["validation"]
        o = _cli("modeling_observe", "--kwargs-json", anchor)
        assert o["biz"]["combo_key"] == "RF|passthrough|passthrough", o["biz"].get("combo_key")
        assert o["biz"]["models"]["RF|passthrough|passthrough"]["u_inner"] is not None
        print(f"瘦装会话远程冒烟成立：donor={donor[:8]} steps=1 u_inner 在账（{repo.name}）")


if __name__ == "__main__":
    main()
