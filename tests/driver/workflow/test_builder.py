"""Builder 模块测试"""

from pathlib import Path

from mflowy.driver.builder import Builder
from mflowy.driver.config import StepType
from mflowy.driver.context import Context
from mflowy.driver.handler import _REGISTRY


def _dummy_handler(ctx):
    pass


def _print_mermaid_dag(workflow):
    """输出Mermaid格式的DAG图"""
    print("\n=== DAG Structure (Mermaid) ===")
    print("```mermaid")
    print("flowchart TD")

    nodes = set()
    edges = []
    visited = set()

    def collect_dag(task: Context):
        if task in visited:
            return
        visited.add(task)

        node_id = task.conf.name.replace("_", "").replace("-", "")
        nodes.add((node_id, task.conf.name, task.id, len(task._prevs)))

        for prev in task._prevs:
            prev_id = prev.conf.name.replace("_", "").replace("-", "")
            edges.append((prev_id, node_id))

        for next_task in task._nexts:
            collect_dag(next_task)

    for start_task in workflow.starts:
        collect_dag(start_task)

    for node_id, name, task_id, prevs_count in nodes:
        print(f"    {node_id}[{name}<br/><sub>id={task_id}, prevs={prevs_count}</sub>]")

    for src, dst in edges:
        print(f"    {src} --> {dst}")

    print("```")
    print()


class TestBuilder:
    def setup_method(self):
        from mflowy.driver.discover import ensure_discovered

        ensure_discovered()  # 惰性扫描：先补全真实注册表，再覆盖 dummy
        self.original_registry = _REGISTRY.copy()
        # 注册足够多的模块名以覆盖测试中使用的 YAML 配置
        _module_names = ["csv", "common_filter", "standard_scaler", "XGBoost", "correlation_heatmap", "test"]
        for step_type in StepType:
            for name in _module_names:
                _REGISTRY[(step_type, name)] = _dummy_handler

    def teardown_method(self):
        _REGISTRY.clear()
        _REGISTRY.update(self.original_registry)

    def test_builder_builds_task_tree_with_prevs(self):
        """测试 Builder 构建 Task 树并建立 prevs 链"""
        config_content = """
workflow:
  steps:
    - name: load_data
      type: load
      module: csv
    - name: clean_data
      type: clean
      module: common_filter
      steps:
        - name: filter_columns
          type: x_transformer
          module: standard_scaler
    - name: transform_data
      type: x_transformer
      module: standard_scaler
"""
        task_yaml = Path("test_config.yaml")
        task_yaml.write_text(config_content)

        try:
            builder = Builder(str(task_yaml))
            workflow = builder.build()

            _print_mermaid_dag(workflow)

            assert len(workflow.starts) == 1

            start_task = workflow.starts[0]
            assert start_task.conf.name == "load_data"
            assert len(start_task._prevs) == 0
            assert len(start_task._nexts) == 1

            clean_task = start_task._nexts[0]
            assert clean_task.conf.name == "clean_data"
            assert len(clean_task._prevs) == 1
            assert clean_task._prevs[0] == start_task
            # filter_columns (clean_data 子节点) 与 transform_data (顶层) 均为无下游 model 的 x_transformer 孤儿，按规则剪除
            assert len(clean_task._nexts) == 0

        finally:
            task_yaml.unlink()

    def test_builder_injects_env_as_direct_template_variables(self, tmp_path):
        """测试 Builder 将 env 字典展开为直接模板变量"""
        config_content = """
workflow:
  name: "测试工作流"
  steps:
    - name: load_data
      type: load
      module: csv
      params:
        source: "{{ workfolder }}/data.csv"
        output_dir: "{{ output_dir }}"
"""
        task_yaml = tmp_path / "test_config.yaml"
        task_yaml.write_text(config_content)

        try:
            builder = Builder(
                str(task_yaml),
                env={"workfolder": "/data", "output_dir": "/output"},
            )
            conf = builder.config
            step = conf.workflow.steps[0]

            assert step.params["source"] == "/data/data.csv"
            assert step.params["output_dir"] == "/output"
        finally:
            pass

    def test_builder_env_override_replaces_value(self, tmp_path):
        """测试 CLI -e 覆盖 env 文件中的同名变量"""
        config_content = """
workflow:
  steps:
    - name: step1
      type: load
      module: csv
      params:
        path: "{{ output_dir }}/result.csv"
"""
        task_yaml = tmp_path / "test_config.yaml"
        task_yaml.write_text(config_content)

        try:
            builder = Builder(
                str(task_yaml),
                env={"output_dir": "/default", "extra": "val"},
            )
            step = builder.config.workflow.steps[0]
            assert step.params["path"] == "/default/result.csv"
        finally:
            pass
