"""log_load_data_fingerprint：next() 后把更新过的 workflow_tags 补写到当前 active run"""

from unittest.mock import patch

from mflowy.builtin_plugins.middlewares import log_load_data_fingerprint
from mflowy.driver.config import StepConf
from mflowy.driver.context import Context
from mflowy.utils.mlflow import set_workflow_tags, workflow_tags


def test_patches_active_run_with_updated_tags():
    set_workflow_tags({"mflowy.modeling_yaml_sha256": "yamlhash"})
    ctx = Context(StepConf(name="加载", type="load", module="csv"), [])

    def next_handler(c, **kw):
        set_workflow_tags({**workflow_tags(), "mflowy.data_sha256": "datahash"})  # 模拟 fn 内 set_data_fingerprint
        return "df"

    with patch("mflowy.builtin_plugins.middlewares.log_load_data_fingerprint.set_tags") as m:
        result = log_load_data_fingerprint(ctx, next_handler)

    assert result == "df"
    m.assert_called_once_with({"mflowy.modeling_yaml_sha256": "yamlhash", "mflowy.data_sha256": "datahash"})
    set_workflow_tags(None)


def test_no_tags_no_call():
    set_workflow_tags(None)
    ctx = Context(StepConf(name="加载", type="load", module="csv"), [])
    with patch("mflowy.builtin_plugins.middlewares.log_load_data_fingerprint.set_tags") as m:
        log_load_data_fingerprint(ctx, lambda c, **kw: "df")
    m.assert_not_called()
