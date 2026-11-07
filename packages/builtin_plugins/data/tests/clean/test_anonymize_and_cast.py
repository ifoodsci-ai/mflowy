"""clean 新四算子契约（#116）：mask_cols / hash_cols / cast_cols / to_category。

- 脱敏判据单源 = detect.rules.mask_text（INV-2 同源——evidence 样本与修复面同一把尺）
- hash 假名化保留可连接性（同值同哈希）；cast 转换失败 fail-loud（不静默造 NaN）
- 全员返回新 df、不改输入（INV-6——词表参数化钉测见 test_smoke.py）
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.clean.anonymize import hash_cols, mask_cols
from mflowy.data.clean.cast_cols import cast_cols, to_category

DF = pd.DataFrame(
    {
        "sid": ["ORD-000123", "ORD-000456", "ORD-000789"],
        "phone": ["13812345678", "13987654321", ""],
        "num_str": ["1.5", "2", "3"],
        "cat": ["a", "b", "a"],
    }
)


# ── mask_cols ───────────────────────────────────────────────────────────────


def test_mask_cols_uses_shared_mask_text():
    """首尾保留 + 中间遮蔽——与 detect 证据样本同一把尺（mask_text 单源）。"""
    out = mask_cols(load=DF, cols=["sid"])
    assert list(out["sid"]) == ["OR******23", "OR******56", "OR******89"]
    assert list(out["phone"]) == list(DF["phone"])  # 未点名列不动


def test_mask_cols_custom_keep_and_char():
    out = mask_cols(load=DF, cols=["phone"], keep_head=3, keep_tail=4, mask_char="#")
    assert out["phone"].iloc[0] == "138####5678"


def test_mask_cols_fail_loud():
    with pytest.raises(ValueError, match="不在数据框中"):
        mask_cols(load=DF, cols=["ghost"])
    with pytest.raises(ValueError, match="cols 必填"):
        mask_cols(load=DF, cols=[])


# ── hash_cols ───────────────────────────────────────────────────────────────


def test_hash_cols_pseudonymization_keeps_linkage():
    """同值同哈希（可连接性保留）；salt 变换 → 全列换名。"""
    df = pd.DataFrame({"sid": ["u1", "u2", "u1"], "x": [1, 2, 3]})

    out = hash_cols(load=df, cols=["sid"])

    assert out["sid"].iloc[0] == out["sid"].iloc[2]  # 同值同哈希
    assert out["sid"].iloc[0] != out["sid"].iloc[1]
    assert out["sid"].iloc[0].startswith("u1") is False  # 原文不残留
    salted = hash_cols(load=df, cols=["sid"], salt="pepper")
    assert salted["sid"].iloc[0] != out["sid"].iloc[0]


def test_hash_cols_algo_length_and_guards():
    out = hash_cols(load=DF, cols=["sid"], algo="md5", length=12)
    assert (out["sid"].str.len() == 12).all()
    with pytest.raises(ValueError, match="algo"):
        hash_cols(load=DF, cols=["sid"], algo="bogus")
    with pytest.raises(ValueError, match="length"):
        hash_cols(load=DF, cols=["sid"], length=4)  # 过短截断（≥8 下限）
    with pytest.raises(ValueError, match="不在数据框中"):
        hash_cols(load=DF, cols=["ghost"])


def test_hash_cols_preserves_missing():
    df = pd.DataFrame({"sid": ["u1", None]})
    out = hash_cols(load=df, cols=["sid"])
    assert pd.isna(out["sid"].iloc[1])  # 缺失不伪造哈希


# ── cast_cols ───────────────────────────────────────────────────────────────


def test_cast_cols_numeric_and_str():
    out = cast_cols(load=DF, cols=["num_str"], target_dtype="float")
    assert out["num_str"].dtype.kind == "f" and out["num_str"].iloc[0] == 1.5

    back = cast_cols(load=out, cols=["num_str"], target_dtype="str")
    assert back["num_str"].map(type).eq(str).all()


def test_cast_cols_fail_loud_on_unparseable():
    """转换失败 fail-loud——不静默 coerce 造 NaN（红线 1：宁可拒绝，绝不静默改写）。"""
    with pytest.raises((ValueError, TypeError), match="num_str"):
        cast_cols(load=DF.assign(num_str=["1.5", "2%", "3"]), cols=["num_str"], target_dtype="float")
    with pytest.raises(ValueError, match="target_dtype"):
        cast_cols(load=DF, cols=["num_str"], target_dtype="decimal")
    with pytest.raises(ValueError, match="不在数据框中"):
        cast_cols(load=DF, cols=["ghost"], target_dtype="float")


# ── to_category ─────────────────────────────────────────────────────────────


def test_to_category_dtype_and_order():
    out = to_category(load=DF, cols=["cat"])
    assert str(out["cat"].dtype) == "category"
    assert list(out["cat"].cat.categories) == ["a", "b"]

    ordered = to_category(load=DF, cols=["cat"], ordered=True)
    assert ordered["cat"].cat.ordered


def test_to_category_fail_loud():
    with pytest.raises(ValueError, match="不在数据框中"):
        to_category(load=DF, cols=["ghost"])


# ── INV-6：返回新 df 不改输入 ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (mask_cols, {"cols": ["sid"]}),
        (hash_cols, {"cols": ["sid"]}),
        (cast_cols, {"cols": ["num_str"], "target_dtype": "float"}),
        (to_category, {"cols": ["cat"]}),
    ],
    ids=["mask", "hash", "cast", "category"],
)
def test_ops_return_new_df_input_unchanged(fn, kwargs):
    """clean 原子返回新 df、输入不变（INV-6——重放与回滚的安全地基）。"""
    df = DF.copy()
    out = fn(load=df, **kwargs)

    assert out is not df
    pd.testing.assert_frame_equal(df, DF)  # 输入零突变
    assert isinstance(out, pd.DataFrame)


def test_ops_keep_other_columns_intact():
    """单一副作用：只动点名列。"""
    out = mask_cols(load=DF, cols=["sid"])
    np.testing.assert_array_equal(out["num_str"].to_numpy(), DF["num_str"].to_numpy())
    assert list(out.columns) == list(DF.columns)
