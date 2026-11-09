"""clean 词表参数化冒烟（#119）：遍历运行期 clean 词表（discover 现算）——

- 每算子用最小合法参数跑通（参数面闭包：param_schema 必填字段 ⊆ 夹具参数——新算子
  新必填参数即本测红，强制补夹具）
- 返回新 df、输入零突变（INV-6——重放与回滚的安全地基；空 cols 被拒不静默容忍，
  INV-7 dissolve）
"""

import pandas as pd
import pytest
from mflowy.driver.discover import get_node, namespace_nodes

VOCAB = sorted(namespace_nodes("clean"))  # 现算——词表 = 已安装能力

# 最小合法参数夹具（必填参数的取值——按 param_schema 必填面闭包）
_PARAMS: dict[str, dict | None] = {
    "drop_cols": {"cols": ["cat"]},
    "drop_duplicate": {},
    "drop_missing_rows": {"cols": ["hole"]},
    "drop_outlier_iqr": {"cols": ["x"]},
    "drop_outlier_z": {"cols": ["x"]},
    "drop_outside": {"cols": ["x"], "lo": -100.0, "hi": 100.0},
    "fill_bfill": {"cols": ["x"]},
    "fill_const": {"cols": ["cat"], "value": "Unknown"},
    "fill_ffill": {"cols": ["x"]},
    "fill_interpolate": {"cols": ["x"]},
    "fill_mean": {"cols": ["x"]},
    "fill_median": {"cols": ["x"]},
    "fill_mode": {"cols": ["cat"]},
    "cap_iqr": {"cols": ["x"]},
    "cap_z": {"cols": ["x"]},
    "range_clip": {"ranges": {"x": [-100.0, 100.0]}},
    "add_missing_flag": {"cols": ["x"]},
    "normalize_units": {"columns": {"pct": {"to": "%", "factors": {"%": 1}}}},  # #151
    "mask_cols": {"cols": ["cat"]},
    "hash_cols": {"cols": ["cat"]},
    "cast_cols": {"cols": ["num_str"], "target_dtype": "float"},
    "to_category": {"cols": ["cat"]},
    "python": None,  # 条件型逃生舱——source 指向临时脚本（parametrize 内单独构造）
}


def _smoke_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1.0, None, 3.0, 4.0, 5.0],
            "cat": ["a", None, "b", "a", "c"],
            "num_str": ["1.5", "2", "3", "4", "5"],
            "pct": ["10.0%", "20%", "30%", "40%", "50%"],
            "hole": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture(scope="module")
def py_script(tmp_path_factory):
    p = tmp_path_factory.mktemp("pyclean") / "clean.py"
    p.write_text(
        "import pandas as pd\n\n\ndef clean(df: pd.DataFrame) -> pd.DataFrame:\n    return df.copy()\n",
        encoding="utf-8",
    )
    return str(p)


@pytest.mark.parametrize("op", VOCAB, ids=str)
def test_clean_op_runs_with_min_params_returns_new_df(op, py_script):
    """词表成员冒烟：最小合法参数跑通 + 新 df + 输入不变（INV-6）+ 参数面闭包。"""
    node = get_node(f"clean__{op}")
    params = _PARAMS.get(op)
    if op == "python":
        params = {"source": py_script}
    assert params is not None, f"词表成员 {op} 缺最小参数夹具（本测强制补 _PARAMS）"

    # 参数面闭包：冻结面必填字段（默认位 = Ellipsis）⊆ 夹具参数（新必填参数即红）
    required = {name for name, (_, default) in (node.param_fields or {}).items() if default is Ellipsis}
    missing = required - set(params)
    assert not missing, f"{op} 必填参数 {sorted(missing)} 不在冒烟夹具"

    df = _smoke_df()
    df_before = df.copy(deep=True)
    out = node.fn(load=df, clean=None, **params)

    assert isinstance(out, pd.DataFrame)
    assert out is not df  # 新对象（原子不返回输入引用）
    pd.testing.assert_frame_equal(df, df_before)  # 输入零突变（INV-6）


def test_vocab_fixture_covers_whole_vocab():
    """夹具表覆盖词表全集（discover 现算对照——卸载/新增即红）。"""
    assert set(VOCAB) == set(_PARAMS), sorted(set(VOCAB) ^ set(_PARAMS))
