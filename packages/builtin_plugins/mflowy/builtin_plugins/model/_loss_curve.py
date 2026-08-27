import logging

import pandas as pd
from mflowy.utils import mlflow as mlflow_util

from .types import ModelLoader, TrainableModel

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
            raise ValueError("无 loss 曲线数据（未训练或 fit 未传 eval_set）")

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


def plot_loss_curve(output: ModelLoader):
    # 聚合fold训练loss数据 (fold, type, iteration, loss)
    loss_df_list = []
    metric_name = "Loss"
    for fold_idx, model in enumerate(output.models):
        assert isinstance(model, TrainableModel)

        try:
            lc = model.get_loss_curve()
        except Exception as e:
            logger.warning(f"跳过 fold {fold_idx} 损失曲线: {e}")
            return

        assert set(lc.columns) == {"type", "iteration", "loss"}, f"loss curve 列不匹配: {lc.columns}"
        metric_name = lc.attrs.get("metric_name") or metric_name
        lc = lc.assign(fold=fold_idx)
        loss_df_list.append(lc)
    lc_df = pd.concat(loss_df_list, ignore_index=True)
    lc_df.attrs["metric_name"] = metric_name
    mlflow_util.log_table(lc_df, "loss_curve.parquet")
    # 绘图
    from mflowy.builtin_plugins.middlewares import log_figure
    from mflowy.builtin_plugins.plots.base import DPI
    from mflowy.builtin_plugins.plots.model_evaluation.loss_curve import loss_curve

    fig = loss_curve(lc_df)
    log_figure(fig, "loss_curve.png", DPI)

    logger.info("损失曲线已绘制: loss_curve.png, loss_curve.json")
