"""匿名化家族原子：mask_cols（脱敏遮蔽）/ hash_cols（假名化哈希）。

脱敏判据单源 = ``detect.rules.mask_text``（INV-2 同源——检测证据样本与修复面同一把
尺）；hash 同值同哈希保留可连接性（join 需求经卡 fk 先验可见），salt 可配。
"""

import hashlib
from typing import Annotated, Literal

import pandas as pd
from mflowy.data.detect.rules import mask_text
from mflowy.data.middlewares import df_diff
from mflowy.driver.node import node

from ._atom import require_cols

_HASH_ALGOS = ("sha256", "sha1", "md5")
_MIN_HASH_LEN = 8  # 截断下限：过短截断的碰撞风险不可静默接受


@node(df_diff, namespace="clean")
def mask_cols(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要脱敏的列（显式清单——首尾保留、中间遮蔽）"],
    keep_head: Annotated[int, "首部保留字符数"] = 2,
    keep_tail: Annotated[int, "尾部保留字符数"] = 2,
    mask_char: Annotated[str, "遮蔽字符（单字符）"] = "*",
) -> pd.DataFrame:
    """按首尾保留字符遮蔽列值（脱敏）——suspected_id/pii 的非删除处置档。"""
    df = clean if clean is not None else load
    cols = require_cols(df, cols)
    if len(mask_char) != 1:
        raise ValueError(f"mask_char 须为单字符，得到 {mask_char!r}")
    if keep_head < 0 or keep_tail < 0:
        raise ValueError(f"keep_head/keep_tail 须非负，得到 {keep_head}/{keep_tail}")
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(
            lambda v: mask_text(v, keep_head=keep_head, keep_tail=keep_tail, mask_char=mask_char) if pd.notna(v) else v
        )
    return out


@node(df_diff, namespace="clean")
def hash_cols(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要假名化的列（显式清单）"],
    algo: Annotated[Literal["sha256", "sha1", "md5"], "哈希算法"] = "sha256",
    length: Annotated[int | None, "截断长度（None=全摘要；下限 8——过短碰撞风险 fail-loud）"] = None,
    salt: Annotated[str, "盐值（默认空；换盐 = 全列换名）"] = "",
) -> pd.DataFrame:
    """列值哈希化（假名化）：同值同哈希保留可连接性；缺失不伪造哈希。"""
    df = clean if clean is not None else load
    cols = require_cols(df, cols)
    if algo not in _HASH_ALGOS:
        raise ValueError(f"algo 须为 {list(_HASH_ALGOS)}，得到 {algo!r}")
    if length is not None and length < _MIN_HASH_LEN:
        raise ValueError(f"length 截断下限 {_MIN_HASH_LEN}（过短碰撞风险），得到 {length}")

    def _hash(v) -> str:
        return hashlib.new(algo, f"{salt}{v}".encode("utf-8")).hexdigest()[:length]

    out = df.copy()
    for c in cols:
        out[c] = out[c].map(lambda v: _hash(v) if pd.notna(v) else v)
    return out
