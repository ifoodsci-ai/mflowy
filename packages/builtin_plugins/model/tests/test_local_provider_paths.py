"""LocalJobProvider 数据路径的 <py_path>:<func> 处理（e543628f 回归）。

predict / inverse_optimization 的 data 可带 :func 后缀（经 load.file → python_loader），
exists/set_task_dir 必须作用在拆掉后缀的路径上，报错信息不得包含 :func 残留。
#224 P4：predict / inverse_optimization 成功链正路径（此前仅错误路径，正路径只有
人肉 make e2e 兜底）。
"""

import asyncio

import pytest
from mflowy.model.job_provider.local import LocalJobProvider


def test_predict_missing_py_func_reports_clean_path(tmp_path):
    missing = tmp_path / "missing.py"
    with pytest.raises(FileNotFoundError, match=str(missing)) as ei:
        asyncio.run(LocalJobProvider().predict(data=f"{missing}:load", model="abc123"))
    assert "FileNotExisted" in str(ei.value)
    assert ":load" not in str(ei.value)


def test_inverse_optimization_missing_py_func_reports_clean_path(tmp_path):
    missing = tmp_path / "missing.py"
    with pytest.raises(FileNotFoundError, match=str(missing)) as ei:
        asyncio.run(LocalJobProvider().inverse_optimization(data=f"{missing}:load", model="abc123"))
    assert "FileNotExisted" in str(ei.value)
    assert ":load" not in str(ei.value)


def test_predict_plain_missing_path_still_reports(tmp_path):
    missing = tmp_path / "data.csv"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        asyncio.run(LocalJobProvider().predict(data=str(missing), model="abc123"))


class TestValidateModelArg:
    """model 参数必须是裸 run_id（flavor 全栈退役，module=run_id 旧格式拒绝）"""

    @pytest.mark.parametrize("raw", ["abc123", "  abc123  "])
    def test_valid(self, raw):
        from mflowy.model.job_provider.local import _validate_model_arg

        assert _validate_model_arg(raw) == raw

    @pytest.mark.parametrize("raw", ["", "XGB=", "=abc", "   =abc", "XGB=abc123", "XGB=a=b"])
    def test_invalid(self, raw):
        from mflowy.model.job_provider.local import _validate_model_arg

        with pytest.raises(ValueError):
            _validate_model_arg(raw)


class TestAnchorTaskDir:
    """B3（#73）：_anchor_task_dir 增认 load 族 ``source`` 参数（csv/excel/parquet/python
    的正名）——修复前只认 ``file_path``，load__python 示例结构性无锚（实验创建即
    「请使用绝对路径」）。"""

    @pytest.fixture(autouse=True)
    def _restore_task_dir(self, tmp_path):
        from mflowy.utils.path import reset_task_dir, set_task_dir

        token = set_task_dir(tmp_path / "known" / ".marker")  # set_task_dir 收文件路径锚其目录
        yield
        reset_task_dir(token)

    def _anchor(self, steps_text: str):
        from mflowy.model.job_provider.local import _anchor_task_dir

        _anchor_task_dir(steps_text)
        from mflowy.utils.path import task_dir

        return task_dir()

    def test_python_source_anchors_stripping_func(self, tmp_path):
        steps = f"- load__python:\n    params:\n      source: {tmp_path}/gen.py:load\n"
        assert self._anchor(steps) == tmp_path

    def test_csv_source_anchors_plain_path(self, tmp_path):
        steps = f"- load__csv:\n    params:\n      source: {tmp_path}/data.csv\n"
        assert self._anchor(steps) == tmp_path

    def test_file_path_still_anchors(self, tmp_path):
        steps = f"- load__file:\n    params:\n      file_path: {tmp_path}/data.parquet\n"
        assert self._anchor(steps) == tmp_path

    def test_url_source_not_anchored(self, tmp_path):
        steps = "- load__file:\n    params:\n      file_path: https://example.com/data.csv\n"
        assert self._anchor(steps) == tmp_path / "known"  # 不锚（无本地目录语义）

    def test_no_load_step_not_anchored(self, tmp_path):
        assert self._anchor("- x_y:\n    params:\n      target: t\n") == tmp_path / "known"

    def test_pinned_env_wins_over_scan(self, tmp_path):
        """#95 裁定 B：env 钉住（CLI MFLOWY_TASK_DIR）→ 扫描降级不推导。"""
        from pathlib import Path as _P

        from mflowy.utils.path import pin_task_dir, reset_task_dir, task_dir_pinned

        token = pin_task_dir("/pinned/dir")
        try:
            assert task_dir_pinned() is True
            steps = f"- load__csv:\n    params:\n      source: {tmp_path}/data.csv\n"
            assert self._anchor(steps) == _P("/pinned/dir")
        finally:
            reset_task_dir(token)


class TestPredictInversePositivePath:
    """#224 P4：成功链正路径——tmp gen.py + 真 mlflow（root conftest autouse 已把
    tracking 隔离进本用例 tmp sqlite）落 name='final' 已训模型，断言 JobOutcome.finished
    与节点值契约（py:func 加载 → model__loader 真取回 → predict/search_input 真算）。"""

    @pytest.fixture(autouse=True)
    def _task_dir(self, tmp_path):
        """DAG 中间件的实验工件根锚进 tmp（ensure_experiment 取 task_dir/.mlruns）。"""
        from mflowy.utils.path import reset_task_dir, set_task_dir

        token = set_task_dir(tmp_path / "gen.py")
        yield
        reset_task_dir(token)

    @pytest.fixture
    def trained_run(self, tmp_path):
        import mlflow
        import numpy as np
        import pandas as pd
        from mflowy.model.model._model import build_pipeline
        from mflowy.model.model._random_forest import model_cls
        from mflowy.model.model.types import TASKTYPE
        from mflowy.utils.mlflow import setup

        setup()  # tracking = autouse 注入的本用例 tmp sqlite
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        exp_id = client.create_experiment("p4-positive", artifact_location=(tmp_path / "arts").as_uri())
        X = pd.DataFrame({"f1": np.arange(20, dtype=float), "f2": np.arange(20, dtype=float) * 2})
        y = pd.DataFrame({"target": X.f1 * 3 + 1.0})
        pipe = build_pipeline(
            TASKTYPE.REGRESSION, model_cls, None, {"n_estimators": 30, "random_state": 0, "n_jobs": 1}
        )
        pipe.fit(X, y)
        pipe.set_names(y.columns.tolist())
        from mflowy.utils.mlflow import log_model

        with mlflow.start_run(experiment_id=exp_id) as run:
            mlflow.set_tags({"mflowy.step": "model.RF"})
            # 生产同款（_pipeline.py save 单源）：pickle 格式——skops 默认拒自定义类型
            log_model(pipe, name="final", input_example=X.head(1), serialization_format="pickle")

        gen = tmp_path / "gen.py"
        gen.write_text(
            "import pandas as pd\n"
            "def load() -> pd.DataFrame:\n"
            "    return pd.DataFrame({'f1': [1.0, 2.0, 3.0, 4.0, 5.0], 'f2': [2.0, 4.0, 6.0, 8.0, 10.0]})\n",
            encoding="utf-8",
        )
        return run.info.run_id, gen

    def test_predict_then_inverse_roundtrip(self, trained_run):
        run_id, gen = trained_run

        outcome = asyncio.run(LocalJobProvider().predict(data=f"{gen}:load", model=run_id))
        assert outcome.status == "finished" and outcome.dag_result is not None
        dag = outcome.dag_result
        assert dag.status == "finished", dag.error
        pred = next(r for r in dag.runs if r.node == "predict")
        df = pred.value
        assert list(df.columns) == ["target"] and len(df) == 5
        # 值级下限：y=f1*3+1 单调可拟合（同 #224 P3 容差口径）
        import numpy as np

        np.testing.assert_allclose(df["target"].to_numpy(), [4.0, 7.0, 10.0, 13.0, 16.0], rtol=0.25)

        outcome2 = asyncio.run(
            LocalJobProvider().inverse_optimization(
                data=f"{gen}:load", model=run_id, direction={"target": "maximize"}, n_trials=4, seed=7
            )
        )
        assert outcome2.status == "finished" and outcome2.dag_result is not None
        si = next(r for r in outcome2.dag_result.runs if r.node == "search_input")
        assert len(si.value) >= 1 and "target" in si.value.columns  # complete trials 带目标列
