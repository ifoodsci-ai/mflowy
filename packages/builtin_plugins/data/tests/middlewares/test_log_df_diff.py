"""df_diff 前值回看契约：clean 参数边（最近前置 clean 实例）优先，无前置回退 load。

回归锚点：多步清洗链第 2 步的 before 取第 1 步 clean 的输出（非原始 load）——
绑定序修复前曾按签名序取首个命中（load 恒先）导致对比基线翻转。
"""

from unittest.mock import patch

import pandas as pd
from mflowy.data.middlewares.log_df_diff import df_diff
from mflowy.driver.builder import Builder
from mflowy.driver.handler import handler


def _dag(fake_plugins, tmp_path):
    @handler(df_diff, namespace="load")
    def _csv(**kw) -> pd.DataFrame:
        return pd.DataFrame({"x1": [1, 2], "x2": [3, 4], "y": [0, 1]})

    @handler(df_diff, namespace="clean")
    def _step1(load: pd.DataFrame, clean: pd.DataFrame | None = None) -> pd.DataFrame:
        return load.drop(columns=["x2"])  # (2,3) → (2,2)

    @handler(df_diff, namespace="clean")
    def _step2(load: pd.DataFrame, clean: pd.DataFrame | None = None) -> pd.DataFrame:
        return clean  # 不再变换——diff 的 before/after 均应为 step1 输出

    fake_plugins.setdefault("load", {})["csv"] = _csv
    fake_plugins.setdefault("clean", {})["step1"] = _step1
    fake_plugins.setdefault("clean", {})["step2"] = _step2
    p = tmp_path / "chain.yaml"
    p.write_text(
        "name: df_diff_test\ndescription: ''\nflow:\n"
        "- load__csv:\n    name: L\n"
        "- clean__step1:\n    name: C1\n"
        "- clean__step2:\n    name: C2\n",
        encoding="utf-8",
    )
    return Builder(p).build()


def test_second_clean_step_diffs_against_previous_clean(fake_plugins, tmp_path, fake_tracking, fake_artifact_window):
    """第 2 步 clean 的 before = 第 1 步输出（(2,2)），非原始 load（(2,3)）。"""
    with patch("mlflow.log_dict"):  # 工件侧照常可 patch（fake-tracking 用例约定）
        flow = _dag(fake_plugins, tmp_path).run()
    assert flow.status == "finished"
    by_name = {r.name: r for r in flow.runs}
    assert "(2, 3) → (2, 2)" in by_name["C1"].output  # 第 1 步：原始 → 删列
    assert "(2, 2) → (2, 2)" in by_name["C2"].output  # 第 2 步：上一步输出为基线


def test_first_clean_step_falls_back_to_load(fake_plugins, tmp_path, fake_tracking, fake_artifact_window):
    """第 1 步 clean 无前置 clean → 回退 load 参数边（原始数据面）。"""
    with patch("mlflow.log_dict"):
        flow = _dag(fake_plugins, tmp_path).run()
    by_name = {r.name: r for r in flow.runs}
    assert "→" in by_name["C1"].output  # 有 diff 输出（回退 load 成功）


def test_no_upstream_skips_diff(fake_plugins, tmp_path, fake_tracking):
    """无编译上游（单节点直调形态）→ 跳过 diff，不报错。"""

    @handler(df_diff, namespace="clean")
    def _solo(load: pd.DataFrame | None = None, clean: pd.DataFrame | None = None) -> pd.DataFrame:
        return pd.DataFrame({"a": [1]})

    fake_plugins.setdefault("clean", {})["solo"] = _solo
    p = tmp_path / "solo.yaml"
    p.write_text(
        "name: solo\ndescription: ''\nflow:\n- clean__solo:\n    name: S\n",
        encoding="utf-8",
    )
    flow = Builder(p).build().run()
    assert flow.status == "finished"
    assert "→" not in flow.runs[0].output  # 无前值：无 diff 行
