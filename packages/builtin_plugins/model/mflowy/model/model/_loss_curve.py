import logging

import pandas as pd
from mflowy.utils import mlflow as mlflow_util

from .types import ModelPipeline

logger = logging.getLogger(__name__)


class LossCurveMixin:
    @staticmethod
    def _build_loss_curve_df(train_metrics: dict, val_metrics: dict, *, metric_name: str | None = None) -> pd.DataFrame:
        """从 train/val 指标字典构造 loss curve DataFrame（公共模板）。

        Args:
            train_metrics: dict[metric_name,score_list]
            val_metrics: dict[metric_name,score_list]

        Returns:
            pd.DataFrame:
                - columns: iteration, type, loss
        """
        if not train_metrics:
            raise ValueError("无 loss 曲线数据（未训练或 fit 未传独立 val（eval_X/eval_y））")

        # 取底层模型默认的第一个损失函数评估指标
        default_metric_name = next(iter(train_metrics))
        metric_name = metric_name or default_metric_name
        if metric_name not in train_metrics:
            logger.warning(f"底层模型中没有记录 {metric_name} 损失函数，回退到 {default_metric_name}")
            metric_name = default_metric_name

        train_loss = train_metrics[metric_name]
        val_loss = val_metrics.get(metric_name, [])
        iterations = list(range(len(train_loss)))
        df_train = pd.DataFrame({"iteration": iterations, "type": "Train", "loss": train_loss})
        if val_loss:
            iterations = list(range(len(val_loss)))
            df_val = pd.DataFrame({"iteration": iterations, "type": "Val", "loss": val_loss})
            df = pd.concat([df_train, df_val], ignore_index=True)
            df.attrs["metric_name"] = metric_name
            return df
        return df_train


def plot_loss_curve(pipe: ModelPipeline):
    """最终模型（单条）的 loss 曲线落盘：loss_curve.parquet + loss_curve.png。"""
    import mlflow

    if mlflow.active_run() is None:
        logger.warning("无 active run，跳过损失曲线（副产物族计算+落盘整体随 run——直调形态不白算）")
        return

    est = pipe.estimator
    try:
        lc = est.get_loss_curve()
    except NotImplementedError:
        logger.info(f"{type(est).__name__} 无 loss 曲线（OOB 类模型），跳过")
        return
    except Exception as e:
        logger.warning(f"跳过损失曲线: {e}")
        return

    assert set(lc.columns) == {"type", "iteration", "loss"}, f"loss curve 列不匹配: {lc.columns}"
    metric_name = lc.attrs.get("metric_name") or "Loss"
    lc.attrs["metric_name"] = metric_name
    mlflow_util.log_table(lc, "loss_curve.parquet", graceful=True)
    # 绘图
    from mflowy.data.middlewares import log_figure
    from mflowy.model.plot.model_evaluation.loss_curve import loss_curve
    from mflowy.utils.plot import DPI

    fig = loss_curve(lc)
    log_figure(fig, "loss_curve.png", DPI, graceful=True)

    logger.info("损失曲线已绘制: loss_curve.png, loss_curve.parquet")
