"""statistic.profile 契约：df.info 薄封装——print-only、无 findings、无重 schema 返回。

重统计与规则化 findings 归 univariate；profile 只答"表长什么样"。
"""

import pandas as pd
from mflowy.data.statistic.profile import profile


def test_profile_prints_info_only(capsys):
    """输出 info 形态(形状/dtype/non-null 计数),返回 None,无 findings 行。"""
    out = profile(load=pd.DataFrame({"a": [1.0, 2.0, None], "b": ["x", "y", "z"], "c": [1, 2, 3]}))

    assert out is None
    stdout = capsys.readouterr().out
    assert "Data columns (total 3 columns)" in stdout
    assert "non-null" in stdout and "float64" in stdout  # info 逐列计数与类型
    assert "[findings]" not in stdout  # findings 归 univariate
