"""Builder 模块测试（平铺叙事）：env 模板变量 / options 变换 / 嵌套键 fail-loud。"""

import pytest
from mflowy.driver.builder import Builder


def _dummy_handler(*args, **_):
    pass


class TestBuilderEnv:
    @pytest.fixture(autouse=True)
    def _fill_dummy_plugins(self, fake_plugins):
        fake_plugins.setdefault("load", {})["csv"] = _dummy_handler

    def test_builder_injects_env_as_direct_template_variables(self, tmp_path):
        """Builder 将 env 字典展开为直接模板变量"""
        dag_yaml = tmp_path / "test_config.yaml"
        dag_yaml.write_text(
            """
name: "测试工作流"
flow:
  - load__csv:
      name: load_data
      params:
        source: "{{ workfolder }}/data.csv"
        output_dir: "{{ output_dir }}"
"""
        )
        builder = Builder(str(dag_yaml), env={"workfolder": "/data", "output_dir": "/output"})
        step = next(iter(builder.config.flow[0].values()))
        assert step.params["source"] == "/data/data.csv"
        assert step.params["output_dir"] == "/output"

    def test_builder_env_override_replaces_value(self, tmp_path):
        """CLI -e 覆盖 env 文件中的同名变量"""
        dag_yaml = tmp_path / "test_config.yaml"
        dag_yaml.write_text(
            """
flow:
  - load__csv:
      name: step1
      params:
        path: "{{ output_dir }}/result.csv"
"""
        )
        builder = Builder(str(dag_yaml), env={"output_dir": "/default", "extra": "val"})
        step = next(iter(builder.config.flow[0].values()))
        assert step.params["path"] == "/default/result.csv"


class TestFlatRejects:
    @pytest.fixture(autouse=True)
    def _fill_dummy_plugins(self, fake_plugins):
        fake_plugins.setdefault("load", {})["csv"] = _dummy_handler

    def test_legacy_identity_keys_fail_loud(self, tmp_path):
        """配置体含 namespace/node_name/enabled 退役键 fail-loud——身份已上移为列表项的键"""
        dag_yaml = tmp_path / "legacy.yaml"
        dag_yaml.write_text(
            """
flow:
  - load__csv:
      name: a
      namespace: load
"""
        )
        with pytest.raises(ValueError, match=r"退役键.*namespace"):
            Builder(str(dag_yaml)).build()

    def test_retired_narrative_keys_fail_loud(self, tmp_path):
        """配置体含 flow/branches/stop_on_error 退役键 fail-loud"""
        dag_yaml = tmp_path / "legacy2.yaml"
        dag_yaml.write_text(
            """
flow:
  - load__csv:
      name: a
      stop_on_error: false
"""
        )
        with pytest.raises(ValueError, match=r"退役键.*stop_on_error"):
            Builder(str(dag_yaml)).build()

    def test_non_single_key_item_fails_loud(self, tmp_path):
        """列表项非单键映射（多键/裸串）fail-loud——键即身份"""
        dag_yaml = tmp_path / "bad.yaml"
        dag_yaml.write_text(
            """
flow:
  - load__csv:
      name: a
    extra: 1
"""
        )
        with pytest.raises(ValueError, match="单键映射"):
            Builder(str(dag_yaml)).build()


def test_options_apply_to_each_node(tmp_path, fake_plugins):
    """BuilderOption 逐节点应用（声明序）"""
    fake_plugins.setdefault("clean", {})["drop_missing"] = _dummy_handler
    yaml = tmp_path / "flat.yaml"
    yaml.write_text(
        """
name: t
flow:
  - clean__drop_missing: {name: a}
  - clean__drop_missing: {name: b}
"""
    )

    def mark_clean(node):
        if node.namespace == "clean":
            node.params = {**node.params, "marked": True}
        return node

    wf = Builder(yaml, mark_clean).build()
    cleans = [t for t in wf.nodes if t.conf.namespace == "clean"]
    assert {t.conf.name for t in cleans} == {"a", "b"}
    assert all(t.conf.params.get("marked") is True for t in cleans)
