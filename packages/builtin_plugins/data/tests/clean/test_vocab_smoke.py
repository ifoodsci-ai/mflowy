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
    # 真改名（#149）：python 逃生舱改名列经编译链（df_diff 值比较）不崩的直接钉测——
    # no-op copy 钉不到该缝
    p.write_text(
        "import pandas as pd\n\n\ndef clean(df: pd.DataFrame) -> pd.DataFrame:\n"
        '    return df.rename(columns={"pct": "pct_norm"})\n',
        encoding="utf-8",
    )
    return str(p)


def _min_params(op: str, py_script: str) -> dict:
    """最小合法参数单源（#149 评审）：python 特例与夹具查表一份——两测分持曾致
    守卫失效（or {} 吞 None，缺夹具静默以空参跑）。"""
    if op == "python":  # 表值 None 是约定（source 指向临时脚本）——特例前置过守卫
        return {"source": py_script}
    params = _PARAMS.get(op)
    assert params is not None, f"词表成员 {op} 缺最小参数夹具（本测强制补 _PARAMS）"
    return dict(params)


@pytest.mark.parametrize("op", VOCAB, ids=str)
def test_clean_op_runs_with_min_params_returns_new_df(op, py_script):
    """词表成员冒烟：最小合法参数跑通 + 新 df + 输入不变（INV-6）+ 参数面闭包。"""
    node = get_node(f"clean__{op}")
    params = _min_params(op, py_script)

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


@pytest.mark.parametrize("op", VOCAB, ids=str)
def test_clean_op_smoke_through_middleware_chain(op, py_script, tmp_path, monkeypatch):
    """#149：词表冒烟走真实编译链（load__file + clean__op 两节点流 Builder）——算子 ×
    中间件（df_diff 值比较/mlflow 遥测面）交互缺陷第一时间暴露。直调 fn 的旧形态
    正是改名列崩溃（df_diff × strip_units）长期未暴露的原因；fn 级用例保留（快，
    钉 INV-6/参数面闭包），本测钉编译链形态。"""
    from mflowy.driver.builder import Builder
    from mflowy.driver.config import NodeConf
    from mflowy.driver.serializer import flow_to_yaml
    from mflowy.utils.path import reset_task_dir, set_task_dir

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    params = _min_params(op, py_script)  # 空 dict = 合法最小参数（drop_duplicate）
    csv = tmp_path / "smoke.csv"
    _smoke_df().to_csv(csv, index=False)

    flow = flow_to_yaml(
        (
            {"load__file": NodeConf(node_key="load__file", name="装载", params={"file_path": str(csv)})},
            {f"clean__{op}": NodeConf(node_key=f"clean__{op}", name=f"链路-{op}", params=params)},
        )
    )
    yml = tmp_path / "chain.yaml"
    yml.write_text(f"name: vocab_chain_{op}\ndescription: 词表冒烟编译链\nflow:\n{flow}", encoding="utf-8")

    token = set_task_dir(yml)
    try:
        result = Builder(str(yml)).build().run()
    finally:
        reset_task_dir(token)
    assert result.status == "finished", (op, result.error)
