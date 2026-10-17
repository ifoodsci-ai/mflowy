"""split 步元数据 mlflow 跟踪中间件

记录划分留痕 metric 与 ``split.json`` artifact（inner 模式、outer 比例、数据指纹、
来源工件溯源）。索引/groups 是物化期派生物——构建期只记 spec 所知
（n_rows/test_size/fingerprint）+ workflow_tags 里的 load 工件血缘（data 系）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils import mlflow as mlflow_util
from mflowy.utils.mlflow import workflow_tags

if TYPE_CHECKING:
    from mflowy.model.cross_validation import SplitSpec


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
    print(f"split: {result.inner_cv}, n={result.n_rows}, test_size={test_size}, fingerprint={result.data_fingerprint}")
    return result
