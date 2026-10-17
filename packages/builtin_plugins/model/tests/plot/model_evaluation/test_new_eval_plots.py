"""新增评估图（roc_curve / pr_curve / calibration_curve / bland_altman / 混淆矩阵升级）的数据形状测试。"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from mflowy.model.model.types import TASKTYPE


def _binary_proba_df(perfect: bool = False):
    """构造 (y_name, y_true, class, proba) 概率长表：20 样本二分类。"""
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=20)
    if perfect:
        p1 = y_true * 0.9 + 0.05
    else:
        p1 = rng.uniform(0.2, 0.8, size=20)
    rows = []
    for i in range(20):
        rows.append({"y_name": "t", "y_true": y_true[i], "class": 0, "proba": 1 - p1[i]})
        rows.append({"y_name": "t", "y_true": y_true[i], "class": 1, "proba": p1[i]})
    return pd.DataFrame(rows)


def _pred_df():
    rng = np.random.default_rng(7)
    y = rng.normal(10, 2, size=30)
    return pd.DataFrame({"y_name": "t", "y": y, "y_pred": y + rng.normal(0, 0.5, size=30)})


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


class TestRocCurve:
    def test_returns_curve_df_and_fig(self):
        from mflowy.model.plot.model_evaluation.classification.roc_curve import roc_curve

        df, fig = roc_curve(model_test_probabilities=_binary_proba_df())
        assert {"y_name", "class", "fpr", "tpr", "auc"} <= set(df.columns)
        assert df["auc"].between(0, 1).all()
        assert fig is not None

    def test_single_class_group_skipped(self):
        from mflowy.model.plot.model_evaluation.classification.roc_curve import roc_curve

        df = _binary_proba_df()
        df.loc[df["class"] == 1, "y_true"] = 0  # 正类无正样本
        out, fig = roc_curve(model_test_probabilities=df)
        assert out.empty  # 唯一可画组被跳过


class TestPrCurve:
    def test_returns_curve_df_with_ap(self):
        from mflowy.model.plot.model_evaluation.classification.pr_curve import pr_curve

        df, fig = pr_curve(model_test_probabilities=_binary_proba_df())
        assert {"recall", "precision", "ap"} <= set(df.columns)
        assert df["ap"].between(0, 1).all()


class TestCalibrationCurve:
    def test_perfect_calibration_low_ece(self):
        from mflowy.model.plot.model_evaluation.classification.calibration_curve import calibration_curve

        df, fig = calibration_curve(model_test_probabilities=_binary_proba_df(perfect=True))
        assert {"confidence", "accuracy", "count"} <= set(df.columns)
        assert not df.empty

    def test_points_and_ece_brier(self):
        from mflowy.model.plot.model_evaluation.classification.calibration_curve import _calibration_points

        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, size=100)
        # 连续置信度：正类概率 = y_true*0.6 + U(0.2,0.4)——校准良好且分位数分桶有多个非空桶
        p1 = y_true * 0.6 + rng.uniform(0.2, 0.4, size=100)
        proba = np.stack([1 - p1, p1], axis=-1)
        rows, ece, brier = _calibration_points(y_true, proba, np.array([0, 1]), n_bins=10)
        assert len(rows) >= 5
        assert 0 <= ece <= 1
        assert 0 <= brier <= 1
        # 校准较准的模型 ECE 应偏小
        assert ece < 0.3


class TestBlandAltman:
    def test_returns_stats_df(self):
        from mflowy.model.plot.model_evaluation.regression.bland_altman import bland_altman

        stats, fig = bland_altman(model_test_predictions=(TASKTYPE.REGRESSION, _pred_df()))
        assert {"mean_diff", "sd_diff", "loa_lo", "loa_hi"} <= set(stats.columns)
        assert (stats["loa_lo"] < stats["mean_diff"]).all()
        assert (stats["mean_diff"] < stats["loa_hi"]).all()


class TestConfusionMatrixUpgrade:
    def test_per_class_report(self):
        from mflowy.model.plot.model_evaluation.classification.confusion_matrix import per_class_report

        df = pd.DataFrame(
            {
                "y": [0, 0, 1, 1, 1, 2],
                "y_pred": [0, 1, 1, 1, 0, 2],
            }
        )
        report = per_class_report(df)
        assert list(report.columns) == ["class", "precision", "recall", "f1", "support"]
        by_cls = report.set_index("class")
        assert by_cls.loc[1, "precision"] == pytest.approx(2 / 3)
        assert by_cls.loc[1, "recall"] == pytest.approx(2 / 3)
        assert by_cls.loc[2, "recall"] == 1.0
        assert by_cls.loc[0, "support"] == 2

    def test_normalized_crosstab_vocabulary(self):
        """normalize=true（sklearn 词表）映射到 pandas 的 index——不抛 ValueError"""
        from mflowy.model.plot.model_evaluation.classification.confusion_matrix import confusion_matrix

        df = pd.DataFrame(
            {
                "y_name": ["t"] * 6,
                "y": [0, 0, 1, 1, 1, 2],
                "y_pred": [0, 1, 1, 1, 0, 2],
            }
        )
        report, fig = confusion_matrix(model_test_predictions=(TASKTYPE.CLASSIFICATION, df), normalize="true")
        assert not report.empty
