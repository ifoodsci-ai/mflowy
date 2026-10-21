"""split 步元数据 mlflow 跟踪中间件

记录划分留痕 metric 与 ``split.json``/``split.pkl`` artifact（inner 模式、outer
比例、数据指纹、来源工件溯源、物化划分副产物——#66 节点副产物契约）。索引/groups
是物化期派生物——构建期只记 spec 所知（n_rows/test_size/fingerprint）+ workflow_tags
里的 load 工件血缘（data 系）。split.pkl 是划分身份的载体（run_id 引用，
``load_split`` 节点按引用复用）。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils import mlflow as mlflow_util
from mflowy.utils.mlflow import workflow_tags

if TYPE_CHECKING:
    from mflowy.model.cross_validation import SplitSpec

logger = logging.getLogger(__name__)


def log_cv(ctx: Context, next: Handler) -> SplitSpec:
    # 函数内导入破环：cross_validation（装饰器要本中间件）↔ 本模块（断言要 SplitSpec）
    # ——契约并入单文件后，防环约束从旧 types.py 叶子移交到此处延迟
    import mlflow
    from mflowy.model.cross_validation import SplitSpec

    result = next(ctx)
    assert isinstance(result, SplitSpec)

    test_size = getattr(result.outer, "test_size", None)
    # 来源工件溯源：load 起点的文件级血缘（mflowy.data_* 系 tags）——工件 sha 可事后对盘复验，
    # 与 spec 的行内容指纹（语义守卫）双粒度并存；无 data tags（会话路径/无 load DAG）记 None
    source_artifacts = {
        k.removeprefix("mflowy."): v
        for k, v in (workflow_tags() or {}).items()
        if k.startswith("mflowy.data_sha256") or k.startswith("mflowy.data_file")
    } or None
    meta = {
        "inner_cv": result.inner_cv,
        "inner_splitter": repr(result.inner),
        "outer_splitter": repr(result.outer),
        "test_size": test_size,
        "n_rows": result.n_rows,
        "group_by": result.group_by,
        "data_fingerprint": result.data_fingerprint,
        "data_summary": result.data_summary,
        "source_artifacts": source_artifacts,
    }
    metrics = {"n_rows": result.n_rows}
    if test_size is not None:
        metrics["test_size"] = float(test_size)
    mlflow.log_metrics(metrics)
    mlflow_util.log_dict(meta, "split.json")

    # #66 副产物：物化划分 split.pkl 直写 run 工件树（划分身份 = run_id 引用）。
    # x_y 上游经 ctx 侧信道回看（builder 播种 by_param；会话链调用同形播种）——
    # 绑定序取最近命中（单值形态与注入语义对齐）
    from mflowy.model.x_y import XySpec

    res = getattr(ctx.upstream_by_param("x_y")[-1], "result", None) if ctx.upstream_by_param("x_y") else None
    if isinstance(res, XySpec):
        from mflowy.model.model._split import materialize_split
        from mflowy.utils.mlflow import log_pickle

        log_pickle(materialize_split(result, res), "split.pkl")
    else:
        logger.warning("split.pkl 副产物跳过（x_y 上游不可见——非节点链调用形态无引用载体）")
    print(f"split: {result.inner_cv}, n={result.n_rows}, test_size={test_size}, fingerprint={result.data_fingerprint}")
    return result
