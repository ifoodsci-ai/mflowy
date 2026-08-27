"""测试 statistic.profile handler"""

from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd
import pytest
from mflowy.driver import discover
from mflowy.driver.config import StepConf
from mflowy.driver.context import Context


def _make_ctx(**params):
    conf = StepConf(name="test-inspect", type="statistic", params=params)
    return Context(conf=conf)


def _make_load_prev(df: pd.DataFrame) -> Context:
    prev_conf = StepConf(name="load", type="load", params={"module": "csv"})
    prev_ctx = Context(conf=prev_conf)
    prev_ctx.result = df
    return prev_ctx


class TestProfileHandler:
    def test_schema_per_column(self):
        """每个数据列产生一行 schema，字段包含 name/dtype/missing。"""
        h = discover.get("statistic", "profile")

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        ctx = Context(_make_ctx().conf, prevs=[_make_load_prev(df)])

        with ExitStack() as stack:
            for p in ("mlflow.start_run", "mlflow.set_tag", "mlflow.log_params", "mlflow.log_dict"):
                stack.enter_context(patch(p))
            stack.enter_context(patch("mflowy.utils.mlflow.append_tag"))
            result = h(ctx)

        assert result.shape[0] == 2  # 每列 1 行
        assert set(result.columns) >= {"name", "dtype", "missing", "nunique", "is_id", "is_constant", "top_10"}
        assert result["name"].tolist() == ["a", "b"]

    def test_no_load_prev_raises(self):
        """无 LOAD 前置步骤时 inject_df 抛 PreviousContextNotFoundError。"""
        h = discover.get("statistic", "profile")

        ctx = _make_ctx()

        with ExitStack() as stack:
            for p in ("mlflow.start_run", "mlflow.set_tag", "mlflow.log_params", "mlflow.log_dict"):
                stack.enter_context(patch(p))
            stack.enter_context(patch("mflowy.utils.mlflow.append_tag"))
            with pytest.raises(Exception, match="缺少 load 前置节点"):
                h(ctx)

    def test_logs_stats_to_mlflow(self):
        """log_statistic middleware 将 schema 记录到 mlflow。"""
        h = discover.get("statistic", "profile")

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        ctx = Context(_make_ctx().conf, prevs=[_make_load_prev(df)])

        with ExitStack() as stack:
            mock_log_table = stack.enter_context(patch("mlflow.log_table"))
            for p in ("mlflow.start_run", "mlflow.set_tag", "mlflow.log_params"):
                stack.enter_context(patch(p))
            stack.enter_context(patch("mflowy.utils.mlflow.append_tag"))
            h(ctx)

        mock_log_table.assert_called_once()


class TestProfileSchemaFields:
    """直接调用 profile 函数验证 nunique/is_id/top_10 字段语义"""

    def _profile(self, df):
        from mflowy.builtin_plugins.statistic.profile import profile

        return profile(df).set_index("name")

    def test_nunique_and_is_id(self):
        import numpy as np

        df = pd.DataFrame(
            {
                "uid": ["u1", "u2", "u3", "u4"],  # 字符串全唯一 → ID
                "code": [10, 20, 30, 40],  # int 全唯一 → ID
                "measure": [1.1, 2.2, 3.3, 4.4],  # float 全唯一 → 排除，非 ID
                "cat": ["x", "x", "y", "y"],  # 低基数 → 非 ID
                "empty": [np.nan, np.nan, np.nan, np.nan],  # 全空 → count>0 守卫，非 ID
            }
        )
        schema = self._profile(df)
        assert schema.loc["uid", "nunique"] == 4 and schema.loc["uid", "is_id"]
        assert schema.loc["code", "is_id"]
        assert not schema.loc["measure", "is_id"]
        assert not schema.loc["cat", "is_id"] and schema.loc["cat", "nunique"] == 2
        assert not schema.loc["empty", "is_id"]

    def test_numeric_columns_have_skew_and_kurt(self):
        import numpy as np

        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "right_skewed": np.exp(rng.normal(0, 1, 200)),  # 对数正态 → 右偏
                "symmetric": rng.normal(0, 1, 200),
                "label": ["a", "b"] * 100,
            }
        )
        schema = self._profile(df)
        assert schema.loc["right_skewed", "skew"] > 1  # 高度右偏
        assert abs(schema.loc["symmetric", "skew"]) < 0.5  # 近似对称
        # 非数值列无偏度峰度
        assert pd.isna(schema.loc["label", "skew"])

    def test_top_10_dict_with_others(self):
        # 12 个唯一值 → 前 10 项占比 + others 汇总
        values = [f"v{i}" for i in range(12) for _ in range(12 - i)]  # v0×12 ... v11×1
        df = pd.DataFrame({"c": values})
        schema = self._profile(df)
        top = schema.loc["c", "top_10"]
        assert isinstance(top, dict)
        assert "others" in top and len(top) == 11  # 10 项 + others
        n = len(values)
        assert top["v0"] == round(12 / n, 4)
        assert abs(sum(top.values()) - 1.0) < 1e-6  # 无缺失时占比总和为 1

    def test_top_10_no_others_when_few_uniques(self):
        df = pd.DataFrame({"c": ["a", "a", "b"]})
        top = self._profile(df).loc["c", "top_10"]
        assert "others" not in top
        assert top == {"a": round(2 / 3, 4), "b": round(1 / 3, 4)}
