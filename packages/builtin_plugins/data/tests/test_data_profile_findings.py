"""模板定稿与 findings 编译层退役（#28）：

data_profile = load + statistic.profile；eda = load + univariate + bivariate + plot 族——
detect 步撤出（检测是内层 MDP 的观测基质，永不为外层动作），findings 编译层随退役。
"""

import asyncio

import numpy as np
import pandas as pd
import pytest
from mflowy.data import tools


@pytest.fixture
def csv(tmp_path) -> str:
    n = 50
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"feat_a": rng.normal(size=n), "target": rng.normal(size=n)})
    path = tmp_path / "d.csv"
    pd.concat([df, df.head(5)]).to_csv(path, index=False)
    return str(path)


def test_data_profile_steps_are_load_profile_only(csv, fake_tracking):
    """步构成:仅 load.file + statistic.profile——detect 五步与 plots 撤出。"""
    result = asyncio.run(tools.data_profile(csv))

    assert result.status == "finished"
    assert [r.node for r in result.runs] == ["load__file", "statistic__profile"]
    assert not hasattr(result, "findings") or result.findings is None  # 编译层退役


def test_eda_steps_are_statistic_plots_no_detect(csv, fake_tracking):
    """步构成:load + univariate + bivariate + plot 族——effect_size 与 detect 步撤出。"""
    result = asyncio.run(tools.eda(csv, target="target"))

    assert result.status == "finished"
    nodes = [r.node for r in result.runs]
    assert nodes[0] == "load__file"
    assert "statistic__univariate" in nodes and "statistic__bivariate" in nodes
    assert "statistic__effect_size" not in nodes
    assert any(n.startswith("plot__") for n in nodes)
    assert not any(n.startswith("detect.") for n in nodes)


def test_infer_task_type_tool_retired():
    """infer_task_type 独立工具退役（task 归属建模协议 ProtocolCard，#31）。"""
    assert not hasattr(tools, "infer_task_type_by_statistic")
