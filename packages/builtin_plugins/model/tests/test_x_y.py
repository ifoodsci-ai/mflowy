"""x_y 参数边契约：(load, clean=None) 直连——clean 族优先（末位实例）、无 clean 回退 load。"""

import pandas as pd
from mflowy.driver.builder import Builder
from mflowy.driver.handler import handler
from mflowy.model.x_y import x_y as x_y_fn


def _dag(fake_plugins, tmp_path, with_clean: bool):
    @handler(namespace="load")
    def _csv(**kwargs) -> pd.DataFrame:
        return pd.DataFrame({"x1": [1, 2], "x2": [3, 4], "y": [0, 1]})

    fake_plugins.setdefault("load", {})["csv"] = _csv
    nodes = ["- load__csv:\n    name: L"]

    if with_clean:

        @handler(namespace="clean")
        def _filter(load: pd.DataFrame, **_) -> pd.DataFrame:
            return load.drop(columns=["x2"])

        fake_plugins.setdefault("clean", {})["filter"] = _filter
        nodes.append("- clean__filter:\n    name: C")

    fake_plugins.setdefault("", {})["x_y"] = x_y_fn
    nodes.append("- x_y:\n    name: XY\n    params:\n      target: y")
    p = tmp_path / "chain.yaml"
    p.write_text("name: t\ndescription: ''\nflow:\n" + "\n".join(nodes) + "\n", encoding="utf-8")
    return Builder(p).build()


def test_x_y_prefers_nearest_clean(fake_plugins, tmp_path, fake_tracking):
    """链上有 clean → x_y 消费 clean 组末位输出（删列生效）。"""
    dag = _dag(fake_plugins, tmp_path, with_clean=True)
    dag.run()
    xy = next(t for t in dag.nodes if t.conf.name == "XY")
    X = xy.result.X
    assert list(X.columns) == ["x1"]


def test_x_y_falls_back_to_load(fake_plugins, tmp_path, fake_tracking):
    """链上无 clean → 可选 clean 边零命中省略注入，回退 load 原始输出。"""
    dag = _dag(fake_plugins, tmp_path, with_clean=False)
    dag.run()
    xy = next(t for t in dag.nodes if t.conf.name == "XY")
    X = xy.result.X
    assert list(X.columns) == ["x1", "x2"]
