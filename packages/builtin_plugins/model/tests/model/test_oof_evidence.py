"""OOF 证据四面 + 回归区间半宽（#80/#164）：y / y_pred / y_proba / error 落盘件
与 |error| 分位半宽——BO 消费面（预测范围对带 + 配对反馈分母）与 predict 区间原料。
#218 OOF 产面移进管线：DAG search 行为 run 的 OOF 工件契约（双分支同名对齐）。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.types import TASKTYPE
from mflowy.model.model._pipeline import PREDICT_INTERVAL_ALPHA, oof_evidence_table, oof_half_widths


def _frame(d: dict) -> pd.DataFrame:
    return pd.DataFrame(d)


class TestEvidenceTable:
    def test_regression_four_faces(self):
        """回归：y/y_pred/error 三面（无 proba），error = y − y_pred（残差符号）。"""
        y_true = _frame({"target": [1.0, 2.0, 3.0]})
        y_pred = _frame({"target": [1.5, 1.0, 3.6]})
        ev = oof_evidence_table(y_true, y_pred, None, TASKTYPE.REGRESSION)
        assert list(ev.columns) == [
            "indices",
            "y_target",
            "y_pred_target",
            "error_target",
        ]  # #236：indices 稳定行身份列
        np.testing.assert_allclose(ev["y_target"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ev["y_pred_target"], [1.5, 1.0, 3.6])
        np.testing.assert_allclose(ev["error_target"], [-0.5, 1.0, -0.6])

    def test_classification_misjudgment_indicator_and_proba(self):
        """分类：error = 误判指示 0/1（int）；y_proba_{cls} 单目标平铺（列序 = sorted classes）。"""
        y_true = _frame({"label": ["a", "b", "a"]})
        y_pred = _frame({"label": ["a", "b", "b"]})
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
        ev = oof_evidence_table(y_true, y_pred, y_proba, TASKTYPE.CLASSIFICATION)
        assert list(ev.columns) == [
            "indices",
            "y_label",
            "y_pred_label",
            "error_label",
            "y_proba_a",
            "y_proba_b",
        ]  # #236：indices 列
        assert ev["error_label"].tolist() == [0, 0, 1]
        np.testing.assert_allclose(ev["y_proba_a"], [0.9, 0.2, 0.4])
        np.testing.assert_allclose(ev["y_proba_b"], [0.1, 0.8, 0.6])

    def test_multitarget_proba_namespaced(self):
        """多目标 proba（list 形态）：y_proba_{t}__{cls} 命名空间隔离。"""
        y_true = _frame({"t1": ["a", "b"], "t2": ["x", "y"]})
        y_pred = _frame({"t1": ["a", "b"], "t2": ["x", "y"]})
        y_proba = [np.array([[0.9, 0.1], [0.3, 0.7]]), np.array([[0.8, 0.2], [0.1, 0.9]])]
        ev = oof_evidence_table(y_true, y_pred, y_proba, TASKTYPE.CLASSIFICATION)
        assert "y_proba_t1__a" in ev.columns and "y_proba_t2__y" in ev.columns
        np.testing.assert_allclose(ev["y_proba_t2__y"], [0.2, 0.9])

    def test_row_order_follows_y_true(self):
        """行序 ≡ y_true（log_table 不保索引——与 holdout 副产物同契）。"""
        y_true = _frame({"target": [1.0, 2.0, 3.0, 4.0]})
        y_pred = _frame({"target": [1.0, 2.0, 3.0, 4.0]})
        ev = oof_evidence_table(y_true, y_pred, None, TASKTYPE.REGRESSION)
        assert len(ev) == 4 and (ev["error_target"].to_numpy() == 0).all()


class TestHalfWidths:
    def test_symmetric_errors_quantile(self):
        """|error| 的 1−α 分位：对称误差 ±1 → 半宽恒 1.0。"""
        y_true = _frame({"target": [0.0, 2.0, 0.0, 2.0]})
        y_pred = _frame({"target": [1.0, 1.0, -1.0, 3.0]})
        w = oof_half_widths(y_true, y_pred)
        assert w == {"target": 1.0}

    def test_alpha_constant_pinned(self):
        """α=0.1 常量（#80 裁定单源）——出现真实可调需求再升参数，此钉即移位点。"""
        assert PREDICT_INTERVAL_ALPHA == 0.1

    def test_per_target_widths(self):
        y_true = _frame({"t1": [0.0, 2.0], "t2": [0.0, 2.0]})
        y_pred = _frame({"t1": [1.0, 1.0], "t2": [2.0, 0.0]})
        w = oof_half_widths(y_true, y_pred)
        assert w["t1"] == 1.0 and w["t2"] == 2.0


class TestOofSurfaceInPipeline:
    """#218 OOF 产面移进管线：DAG search 行为（run 窗口内）随 run 产 OOF 证据表
    + 半宽 tag + 任务分组可视化——与 session 直调分支同名工件对齐（remote zip ≡
    local 输出的落点）。"""

    @pytest.fixture(autouse=True)
    def isolate_tracking(self, tmp_path, monkeypatch):
        import mlflow

        uri = f"sqlite:///{tmp_path}/tracking.db"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        mlflow.set_tracking_uri(uri)
        monkeypatch.chdir(tmp_path)

    def test_search_run_carries_oof_artifacts(self):
        """RF search 行为 run：oof_y_pred.parquet（含 indices）+ 回归图族 + 半宽 tag
        + study user_attr（账本回读面）齐备。"""
        import json

        import mlflow
        import numpy as np
        import pandas as pd
        from mflowy.model.cv.split import split as split_fn
        from mflowy.model.model._pipeline import OOF_HALF_WIDTH_TAG
        from mflowy.model.model._split import materialize_split
        from mflowy.model.model.random_forest import RF
        from mflowy.model.model.types import TASKTYPE, MetricName
        from mflowy.model.x_y import XySpec

        rng = np.random.default_rng(4)
        n = 60
        x = rng.normal(size=n)
        df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=n)})
        xy = XySpec(df[["x"]], df[["target"]], TASKTYPE.REGRESSION)
        ts = materialize_split(split_fn(x_y=xy, cv=3), xy)

        with mlflow.start_run(run_name="search:oof") as run:
            evidence = RF(ts, n_trials=2, scoring=MetricName.MAE)
        run_id = run.info.run_id
        arts = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id)}
        assert "oof_y_pred.parquet" in arts  # 证据四面表（indices 列在 TestEvidenceTable 钉）
        assert {"prediction_scatter.png", "residual_scatter.png", "bland_altman.png"} <= arts  # 回归图族
        tags = mlflow.get_run(run_id).data.tags
        half_width = json.loads(tags[OOF_HALF_WIDTH_TAG])
        assert set(half_width) == {"target"} and half_width["target"] > 0  # 回归半宽 tag
        assert evidence["family"] == "RF"  # 搜索证据面不因产面移动改变

    def test_classification_search_run_carries_oof_figures(self):
        """分类 search run：confusion/roc/pr/calibration 图族 + proba 证据列。"""
        import mlflow
        import numpy as np
        import pandas as pd
        from mflowy.model.cv.split import split as split_fn
        from mflowy.model.model._pipeline import OOF_HALF_WIDTH_TAG
        from mflowy.model.model._split import materialize_split
        from mflowy.model.model.random_forest import RF
        from mflowy.model.model.types import TASKTYPE, MetricName
        from mflowy.model.x_y import XySpec

        rng = np.random.default_rng(7)
        n = 80
        x = rng.normal(size=n)
        label = (x > 0).astype(int)
        df = pd.DataFrame({"x": x, "label": label})
        xy = XySpec(df[["x"]], df[["label"]], TASKTYPE.CLASSIFICATION)
        ts = materialize_split(split_fn(x_y=xy, cv=3), xy)

        with mlflow.start_run(run_name="search:oof-cls") as run:
            RF(ts, n_trials=2, scoring=MetricName.F1)
        run_id = run.info.run_id
        arts = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id)}
        assert {"confusion_matrix.png", "roc_curve.png", "pr_curve.png", "calibration_curve.png"} <= arts
        assert OOF_HALF_WIDTH_TAG not in mlflow.get_run(run_id).data.tags  # 分类无区间面
        ev = pd.read_parquet(_download(run_id, "oof_y_pred.parquet"))
        assert "y_proba_0" in ev.columns and "y_proba_1" in ev.columns  # proba 证据列


def _download(run_id: str, name: str) -> str:
    import mlflow

    return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name)
