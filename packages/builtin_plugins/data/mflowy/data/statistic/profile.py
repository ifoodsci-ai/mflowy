"""数据集画像 step handler——df.info 薄封装"""

import pandas as pd
from mflowy.data.middlewares import log_statistic
from mflowy.driver.handler import handler


@handler(log_statistic, namespace="statistic")
def profile(load: pd.DataFrame) -> None:
    """打印 df.info（形状 / 逐列 dtype / non-null 计数）——print-only 观测节点。

    info 非结构化对象、无可规则化——无 findings、无重 schema 返回（列级统计与
    规则化 findings 见 univariate，围绕变量对的关联事实见 bivariate）。
    """
    load.info()
    return None
