"""log_cv 留痕契约：split.json 记 spec 所知 + 来源工件溯源（workflow_tags 的 data 系）。"""

from unittest.mock import patch

from mflowy.driver.config import NodeConf
from mflowy.driver.context import Context
from mflowy.model.cv.types import SplitSpec
from mflowy.model.middlewares import log_cv
from mflowy.utils.mlflow import set_workflow_tags

# 注：log_cv 此处是函数（middlewares.__init__ 再导出遮蔽模块名）——patch 走全局模块属性
from sklearn.model_selection import KFold, ShuffleSplit


def _spec() -> SplitSpec:
    return SplitSpec(inner_cv="2-fold", outer=ShuffleSplit(test_size=0.2), inner=KFold(n_splits=2), n_rows=4)


def _run_mw():
    captured = {}
    with (
        patch("mlflow.log_metrics") as m_metrics,
        patch("mflowy.utils.mlflow.log_dict", side_effect=lambda meta, name: captured.update(name=name, meta=meta)),
    ):
        log_cv(Context(NodeConf(name="划分", node_key="cv__split"), []), lambda c, **kw: _spec())
    return m_metrics, captured


def test_split_json_records_source_artifacts():
    """工件溯源：workflow_tags 的 data_sha256/data_file（含多 load 后缀）进 split.json。"""
    set_workflow_tags(
        {
            "mflowy.data_sha256": "abc",
            "mflowy.data_file": "/tmp/x.csv",
            "mflowy.data_sha256_2": "def",
            "mflowy.data_file_2": "/tmp/y.csv",
            "mflowy.modeling_yaml_sha256": "yamlhash",  # 非数据工件——不收
        }
    )
    _, captured = _run_mw()
    assert captured["name"] == "split.json"
    assert captured["meta"]["source_artifacts"] == {
        "data_sha256": "abc",
        "data_file": "/tmp/x.csv",
        "data_sha256_2": "def",
        "data_file_2": "/tmp/y.csv",
    }
    set_workflow_tags(None)


def test_no_source_artifacts_without_data_tags():
    """无 data 系 tags（会话路径/无 load 的 DAG）→ source_artifacts 为 None，不炸不误记。"""
    set_workflow_tags({"mflowy.modeling_yaml_sha256": "yamlhash"})
    _, captured = _run_mw()
    assert captured["meta"]["source_artifacts"] is None
    set_workflow_tags(None)
