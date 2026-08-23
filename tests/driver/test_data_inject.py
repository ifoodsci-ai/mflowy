"""data_inject 中间件测试

验证数据注入中间件从 ctx 正确提取数据并通过 next(ctx, key=value) 注入 kwargs。
"""

import pandas as pd

from mflowy.driver.config import StepConf, StepType
from mflowy.driver.context import Context

# ========== helpers ==========


def _make_ctx_with_prev(step_type: StepType, module: str, result, **extra_params):
    """构造包含一个前置节点结果的 Context"""
    prev_conf = StepConf(name="prev", type=step_type, module=module)
    prev_ctx = Context(prev_conf)
    prev_ctx.result = result

    ctx_conf = StepConf(name="current", type=StepType.PLOT, module="test", params=extra_params)
    return Context(ctx_conf, prevs=[prev_ctx])


def _make_ctx_with_two_prevs(
    prev1_type,
    prev1_module,
    prev1_result,
    prev2_type,
    prev2_module,
    prev2_result,
    **extra_params,
):
    """构造包含两个前置节点的 Context"""
    prev1_conf = StepConf(name="prev1", type=prev1_type, module=prev1_module)
    prev1_ctx = Context(prev1_conf)
    prev1_ctx.result = prev1_result

    prev2_conf = StepConf(name="prev2", type=prev2_type, module=prev2_module)
    prev2_ctx = Context(prev2_conf)
    prev2_ctx.result = prev2_result

    ctx_conf = StepConf(name="current", type=StepType.PLOT, module="test", params=extra_params)
    return Context(ctx_conf, prevs=[prev1_ctx, prev2_ctx])


# ========== inject_df ==========


class TestInjectDF:
    def test_injects_df_from_prev_clean(self):
        """无 CLEAN 时回退到 LOAD"""
        from mflowy.middlewares.data_inject import inject_df

        df = pd.DataFrame({"a": [1, 2, 3]})
        ctx = _make_ctx_with_prev(StepType.LOAD, "csv", df)
        captured = {}
        inject_df(ctx, lambda c, **kw: captured.update(kw))
        assert "df" in captured
        pd.testing.assert_frame_equal(captured["df"], df)

    def test_prefers_clean_over_load(self):
        """同时有 LOAD 和 CLEAN 时优先 CLEAN"""
        from mflowy.middlewares.data_inject import inject_df

        load_df = pd.DataFrame({"a": [1]})
        clean_df = pd.DataFrame({"a": [2]})
        ctx = _make_ctx_with_two_prevs(
            StepType.LOAD,
            "csv",
            load_df,
            StepType.CLEAN,
            "drop_missing",
            clean_df,
        )
        captured = {}
        inject_df(ctx, lambda c, **kw: captured.update(kw))
        pd.testing.assert_frame_equal(captured["df"], clean_df)


# ========== GetXPreprocessors (累积多个 X_TRANSFORMER 前驱) ==========


class TestGetXPreprocessors:
    def test_no_pre_processor_returns_none(self):
        from mflowy.middlewares.data_inject import GetXPreprocessors

        ctx_conf = StepConf(name="current", type=StepType.MODEL, module="XGBoost")
        ctx = Context(ctx_conf, prevs=[])
        assert GetXPreprocessors(ctx) is None

    def test_collects_multiple_pre_processors(self):
        """多个 X_TRANSFORMER 前驱时累积成 ColumnTransformer"""
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from mflowy.middlewares.data_inject import GetXPreprocessors

        pp1 = ("scaler", StandardScaler(), ["num_col"])
        pp2 = ("encoder", OneHotEncoder(), ["cat_col"])

        prev1_conf = StepConf(name="prev1", type=StepType.X_TRANSFORMER, module="standard_scaler")
        prev1_ctx = Context(prev1_conf)
        prev1_ctx.result = pp1

        prev2_conf = StepConf(name="prev2", type=StepType.X_TRANSFORMER, module="onehot_encoder")
        prev2_ctx = Context(prev2_conf)
        prev2_ctx.result = pp2

        ctx_conf = StepConf(name="current", type=StepType.MODEL, module="XGBoost")
        ctx = Context(ctx_conf, prevs=[prev1_ctx, prev2_ctx])

        ct = GetXPreprocessors(ctx)
        assert ct is not None
        # ColumnTransformer 内部 transformers 列表长度 == 2
        assert len(ct.transformers) == 2


# ========== inject_plot_data (generator) ==========


class TestInjectPlotData:
    def test_injects_plot_data_as_generator(self):
        """inject_plot_data 调用 plot_data(ctx)，将返回的生成器注入 plot_data kwarg"""
        from mflowy.middlewares.data_inject import inject_plot_data

        df = pd.DataFrame({"x": [1, 2, 3]})

        def my_plot_data(ctx):
            yield (df,)

        ctx_conf = StepConf(name="test", type=StepType.PLOT, module="test")
        ctx = Context(ctx_conf, prevs=[])
        captured = {}
        inject_plot_data(my_plot_data)(ctx, lambda c, **kw: captured.update(kw))

        assert "plot_data" in captured
        results = list(captured["plot_data"])
        assert len(results) == 1
        assert isinstance(results[0], tuple)
        pd.testing.assert_frame_equal(results[0][0], df)

    def test_injects_multi_df_tuple(self):
        from mflowy.middlewares.data_inject import inject_plot_data

        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [2]})

        def my_plot_data(ctx):
            yield (df1, df2)

        ctx_conf = StepConf(name="test", type=StepType.PLOT, module="test")
        ctx = Context(ctx_conf, prevs=[])
        captured = {}
        inject_plot_data(my_plot_data)(ctx, lambda c, **kw: captured.update(kw))

        results = list(captured["plot_data"])
        assert len(results) == 1
        assert len(results[0]) == 2

    def test_plot_data_receives_context(self):
        from mflowy.middlewares.data_inject import inject_plot_data

        received_ctx = []

        def my_plot_data(ctx):
            received_ctx.append(ctx)
            yield (pd.DataFrame({"x": [1]}),)

        ctx_conf = StepConf(name="test", type=StepType.PLOT, module="test")
        ctx = Context(ctx_conf, prevs=[])
        captured = {}
        inject_plot_data(my_plot_data)(ctx, lambda c, **kw: captured.update(kw))

        list(captured["plot_data"])

        assert len(received_ctx) == 1
        assert received_ctx[0] is ctx

    def test_plot_data_yields_multiple_times(self):
        from mflowy.middlewares.data_inject import inject_plot_data

        df1 = pd.DataFrame({"x": [1]})
        df2 = pd.DataFrame({"y": [2]})
        df3 = pd.DataFrame({"z": [3]})

        def my_plot_data(ctx):
            yield (df1,)
            yield (df2,)
            yield (df3,)

        ctx_conf = StepConf(name="test", type=StepType.PLOT, module="test")
        ctx = Context(ctx_conf, prevs=[])
        captured = {}
        inject_plot_data(my_plot_data)(ctx, lambda c, **kw: captured.update(kw))

        results = list(captured["plot_data"])
        assert len(results) == 3
