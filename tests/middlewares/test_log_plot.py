"""log_plot 中间件测试

验证 log_plot 兼容 handler 的 4 种返回形态，文件名统一用 ``ctx.conf.module`` + 序号：
- 场景 1：普通函数 return (df, fig) → ``{module}.parquet`` + ``{module}.png``
- 场景 2：普通函数 return ((df1, df2), fig) → ``{module}_0.parquet`` + ``{module}_1.parquet`` + ``{module}.png``
- 场景 3：生成器 yield (df, fig) 多次 → ``{module}_{i}.parquet`` + ``{module}_{i}.png``
- 场景 4：生成器 yield ((df1, df2), fig) 多次 → ``{module}_{i}_{j}.parquet`` + ``{module}_{i}.png``
"""

from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from mflowy.driver.config import StepConf
from mflowy.driver.context import Context


def _make_ctx(**extra_params):
    conf = StepConf(
        name="test_plot",
        type="plot",
        module="test_module",
        params=extra_params,
    )
    return Context(conf)


def _make_fig():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    return fig


# ========== 场景 1：普通函数 return (df, fig) ==========


class TestLogPlotScenario1:
    def test_logs_table_and_figure_for_single_return(self):
        from mflowy.middlewares.log_plot import log_plot

        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        fig = _make_fig()

        def mock_next(ctx, **kw):
            return df, fig

        ctx = _make_ctx()

        with patch("mlflow.log_table") as mock_table, patch("mlflow.log_figure") as mock_fig:
            log_plot(ctx, mock_next)

        mock_table.assert_called_once()
        assert mock_table.call_args[0][1] == "test_module.parquet"
        mock_fig.assert_called_once()
        assert mock_fig.call_args[0][1] == "test_module.png"
        plt.close(fig)


# ========== 场景 2：普通函数 return ((df1, df2), fig) ==========


class TestLogPlotScenario2:
    def test_logs_multiple_dataframes_in_tuple(self):
        from mflowy.middlewares.log_plot import log_plot

        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [2]})
        fig = _make_fig()

        def mock_next(ctx, **kw):
            return (df1, df2), fig

        ctx = _make_ctx()

        with patch("mlflow.log_table") as mock_table, patch("mlflow.log_figure") as mock_fig:
            log_plot(ctx, mock_next)

        assert mock_table.call_count == 2
        call_args = [c[0][1] for c in mock_table.call_args_list]
        assert call_args[0] == "test_module_0.parquet"
        assert call_args[1] == "test_module_1.parquet"
        mock_fig.assert_called_once()
        assert mock_fig.call_args[0][1] == "test_module.png"
        plt.close(fig)


# ========== 场景 3：生成器 yield (df, fig) 多次 ==========


class TestLogPlotScenario3:
    def test_handles_multiple_yields_single_df(self):
        from mflowy.middlewares.log_plot import log_plot

        df1 = pd.DataFrame({"x": [1]})
        df2 = pd.DataFrame({"y": [2]})
        fig1, fig2 = _make_fig(), _make_fig()

        def mock_next(ctx, **kw):
            yield df1, fig1
            yield df2, fig2

        ctx = _make_ctx()

        with patch("mlflow.log_table") as mock_table, patch("mlflow.log_figure") as mock_fig:
            log_plot(ctx, mock_next)

        assert mock_table.call_count == 2
        table_args = [c[0][1] for c in mock_table.call_args_list]
        assert table_args[0] == "test_module_0.parquet"
        assert table_args[1] == "test_module_1.parquet"
        assert mock_fig.call_count == 2
        fig_args = [c[0][1] for c in mock_fig.call_args_list]
        assert fig_args[0] == "test_module_0.png"
        assert fig_args[1] == "test_module_1.png"
        plt.close("all")


# ========== 场景 4：生成器 yield ((df1, df2), fig) 多次 ==========


class TestLogPlotScenario4:
    def test_handles_multiple_yields_multi_df(self):
        from mflowy.middlewares.log_plot import log_plot

        df1a = pd.DataFrame({"a": [1]})
        df1b = pd.DataFrame({"b": [2]})
        df2a = pd.DataFrame({"c": [3]})
        df2b = pd.DataFrame({"d": [4]})
        fig1, fig2 = _make_fig(), _make_fig()

        def mock_next(ctx, **kw):
            yield (df1a, df1b), fig1
            yield (df2a, df2b), fig2

        ctx = _make_ctx()

        with patch("mlflow.log_table") as mock_table, patch("mlflow.log_figure"):
            log_plot(ctx, mock_next)

        assert mock_table.call_count == 4
        call_args = [c[0][1] for c in mock_table.call_args_list]
        assert call_args[0] == "test_module_0_0.parquet"
        assert call_args[1] == "test_module_0_1.parquet"
        assert call_args[2] == "test_module_1_0.parquet"
        assert call_args[3] == "test_module_1_1.parquet"
        plt.close("all")


# ========== 错误处理 ==========


class TestLogPlotErrorHandling:
    def test_closes_figure_on_log_figure_error(self):
        from mflowy.middlewares.log_plot import log_plot

        df = pd.DataFrame({"x": [1]})
        fig = _make_fig()

        def mock_next(ctx, **kw):
            return df, fig

        ctx = _make_ctx()

        with patch("mlflow.log_table"), patch("mlflow.log_figure", side_effect=RuntimeError("MLflow down")):
            with pytest.raises(RuntimeError, match="MLflow down"):
                log_plot(ctx, mock_next)

        assert not plt.fignum_exists(fig.number)

    def test_closes_generator_on_handler_error(self):
        from mflowy.middlewares.log_plot import log_plot

        gen_closed = []

        def mock_next(ctx, **kw):
            def gen():
                try:
                    yield None
                except GeneratorExit:
                    gen_closed.append(True)
                    raise

            return gen()

        ctx = _make_ctx()

        with pytest.raises(TypeError):
            log_plot(ctx, mock_next)

        assert gen_closed


# ========== 参数传递 ==========


class TestLogPlotParams:
    def test_uses_dpi_and_file_type_from_params(self):
        from mflowy.middlewares.log_plot import log_plot

        df = pd.DataFrame({"x": [1]})
        fig = _make_fig()

        def mock_next(ctx, **kw):
            return df, fig

        ctx = _make_ctx(dpi=150, file_type="svg")

        with patch("mlflow.log_table"), patch("mlflow.log_figure") as mock_fig:
            log_plot(ctx, mock_next)

        mock_fig.assert_called_once()
        assert mock_fig.call_args[0][1] == "test_module.svg"
        assert mock_fig.call_args[1]["save_kwargs"]["dpi"] == 150
        plt.close(fig)

    def test_default_dpi_and_file_type(self):
        from mflowy.middlewares.log_plot import log_plot

        df = pd.DataFrame({"x": [1]})
        fig = _make_fig()

        def mock_next(ctx, **kw):
            return df, fig

        ctx = _make_ctx()

        with patch("mlflow.log_table"), patch("mlflow.log_figure") as mock_fig:
            log_plot(ctx, mock_next)

        assert mock_fig.call_args[0][1] == "test_module.png"
        plt.close(fig)
