"""CV 划分元数据 mlflow 跟踪中间件

记录 ``fold_count`` metric 与 ``folds.json`` artifact（每个 fold 的
train/val/test 行号）。

CV handler 返回 ``Iterator[Indices]``，``Indices = (train_idx, val_idx, test_idx)``
均为位置索引 ``np.ndarray``，直接 ``.tolist()`` 序列化即可。
"""

import mlflow

from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils import mlflow as mlflow_util


def log_cv(ctx: Context, next: Handler):
    result = next(ctx)  # Iterator[Indices] 或 list[Indices]
    folds = list(result)  # 物化生成器，便于多次访问 + 长度统计

    folds_meta = {
        str(i): {
            "train_indices": tr.tolist(),
            "val_indices": va.tolist() if va is not None else None,
            "test_indices": te.tolist(),
        }
        for i, (tr, va, te) in enumerate(folds)
    }

    mlflow.log_metrics({"fold_count": len(folds)})
    mlflow_util.log_dict(folds_meta, "folds.json")
    sizes = ", ".join(f"{len(t)}/{len(v) if v is not None else 0}/{len(te)}" for t, v, te in folds)
    print(f"folds: {len(folds)} (train/val/test = {sizes})")
    return folds
