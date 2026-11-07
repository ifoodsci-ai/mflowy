"""M7 通道纯净契约：detect 算子零 print——会话引擎直调不得污染 MCP stdio。

stdout 在 MCP stdio transport 下是 JSON-RPC 通道；检测回显曾经是算子内
``print(report.summary())``，会话路径（无捕获窗口）直漏真实流。算子层去 print 后
本文件钉住两件事：① 裸调（与 ``_detect_all`` 同姿势）零 stdout；② ``summary()``
仍是 token 友好摘要的唯一载体——输出通道由调用方决定（DAG 路径由引擎捕获窗口
归集进 NodeResult.output）。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.detect.collinear import collinear
from mflowy.data.detect.constant import constant
from mflowy.data.detect.duplicates import duplicates
from mflowy.data.detect.errors import ErrorReport
from mflowy.data.detect.id_like import id_like
from mflowy.data.detect.missing import missing
from mflowy.data.detect.outliers import outliers
from mflowy.data.detect.range_violation import range_violation
from mflowy.data.detect.stringly import stringly

OPERATORS = [missing, duplicates, outliers, constant, stringly, id_like, collinear, range_violation]

# 声明式算子的必填先验参数（其余算子零参数直调）
_OP_KWARGS = {range_violation: {"ranges": {"x1": [-3.0, 3.0]}}}


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """五算子各至少一缺陷：缺失/重复/离群/单位污染/共线。"""
    n = 60
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=n)
    x1[0] = 40.0  # 注入 zscore 离群
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": 2.0 * x1 + 0.01 * rng.normal(size=n),  # 与 x1 强共线
            "hole": np.where(rng.random(n) < 0.3, np.nan, 1.0),
            "const": [3.0] * n,  # 常量死列（constant 维度）
            "pct": [f"{v:.1f}%" for v in rng.uniform(0, 100, n)],
            "sid": [f"s{i:03d}" for i in range(n)],
        }
    )
    return pd.concat([df, df.head(5)]).reset_index(drop=True)  # 5 条重复行


# id_like 判据全唯一，重复行破坏一切列的全唯一——喂去重视图（其余算子吃重复）
_DEDUP_OPS = {id_like}


@pytest.mark.parametrize("op", OPERATORS, ids=lambda f: f.__name__)
def test_operator_direct_call_prints_nothing(op, dirty_df, capsys):
    """裸调（与 _detect_all 同姿势）零 stdout——MCP stdio 通道纯净。"""
    df = dirty_df.drop_duplicates().reset_index(drop=True) if op in _DEDUP_OPS else dirty_df
    report = op(load=df, **_OP_KWARGS.get(op, {}))

    assert capsys.readouterr().out == ""
    assert isinstance(report, ErrorReport)
    assert report.errors  # 夹具保证每算子都有缺陷可检，防止空跑假绿


def test_summary_remains_the_token_friendly_carrier(dirty_df, capsys):
    """print 退役，摘要载体仍在——需要人读回显的调用方调 summary()。"""
    report = missing(load=dirty_df)

    assert "missing" in report.summary()
    assert "ErrorReport" in report.summary()
    assert capsys.readouterr().out == ""
