"""测试各种DAG拓扑场景"""

from pathlib import Path

from mflowy.driver.builder import Builder
from mflowy.driver.config import StepType
from mflowy.driver.handler import _REGISTRY
from mflowy.driver.workflow import Workflow


def _dummy_handler(ctx):
    return None


# 各步骤类型的有效模块名
_MODULES = {
    StepType.LOAD: "csv",
    StepType.CLEAN: "common_filter",
    StepType.X_TRANSFORMER: "standard_scaler",
    StepType.CROSS_VALIDATE: "simple_cv",
    StepType.MODEL: "XGBoost",
    StepType.PLOT: "correlation_heatmap",
}


def _print_dag(workflow, title="DAG Structure"):
    """打印DAG结构"""
    print(f"\n=== {title} ===")
    nodes = set()
    edges = []
    visited = set()

    def collect(task, depth=0):
        if task in visited:
            return
        visited.add(task)
        node_id = task.conf.name.replace("_", "").replace("-", "")
        nodes.add(node_id)
        for prev in task._prevs:
            prev_id = prev.conf.name.replace("_", "").replace("-", "")
            edges.append((prev_id, node_id))
        for next_task in task._nexts:
            collect(next_task, depth + 1)

    for start in workflow.starts:
        collect(start)

    print("```mermaid")
    print("flowchart TD")
    for node_id in nodes:
        print(f"    {node_id}[{node_id}]")
    for src, dst in edges:
        print(f"    {src} --> {dst}")
    print("```\n")


class TestDAGScenarios:
    """测试各种DAG拓扑场景"""

    def setup_method(self):
        from mflowy.driver.discover import ensure_discovered

        ensure_discovered()  # 惰性扫描：先补全真实注册表，再覆盖 dummy
        self.original_registry = _REGISTRY.copy()
        # 注册测试中使用的模块名
        _module_names = ["csv", "common_filter", "standard_scaler", "XGBoost", "correlation_heatmap", "test"]
        for step_type in StepType:
            for name in _module_names:
                _REGISTRY[(step_type, name)] = _dummy_handler

    def teardown_method(self):
        _REGISTRY.clear()
        _REGISTRY.update(self.original_registry)

    def _build_yaml(self, steps_yaml):
        config = f"workflow:\n  steps:\n{steps_yaml}"
        return config

    def test_scenario_1_1_serial(self):
        """场景1：1-1 串行"""
        config = self._build_yaml("""
    - name: A
      type: load
      module: csv
    - name: B
      type: clean
      module: common_filter
    - name: C
      type: x_transformer
      module: standard_scaler
""")
        task_yaml = Path("tests/test_scenario_1.yaml")
        task_yaml.write_text(config)
        try:
            builder = Builder(str(task_yaml))
            workflow = builder.build()

            assert len(workflow.starts) == 1
            a = workflow.starts[0]
            assert a.conf.name == "A"
            assert len(a._nexts) == 1

            b = a._nexts[0]
            assert b.conf.name == "B"
            assert len(b._prevs) == 1
            assert b._prevs[0] == a
            # C (x_transformer) 为无下游 model 的孤儿，按规则剪除
            assert len(b._nexts) == 0
        finally:
            task_yaml.unlink()

    def test_scenario_1_n_parallel(self):
        """场景2：1-N 并行分支"""
        config = self._build_yaml("""
    - name: A
      type: load
      module: csv
      branches:
        - name: A1
          type: clean
          module: common_filter
        - name: A2
          type: x_transformer
          module: standard_scaler
        - name: A3
          type: plot
          module: correlation_heatmap
""")
        task_yaml = Path("tests/test_scenario_2.yaml")
        task_yaml.write_text(config)
        try:
            workflow = Builder(str(task_yaml)).build()
            assert len(workflow.starts) == 1
            a = workflow.starts[0]
            # A2 (x_transformer) 为无子节点的孤儿分支，按规则剪除；剩 A1、A3
            assert len(a._nexts) == 2
            assert {b.conf.name for b in a._nexts} == {"A1", "A3"}
            for branch in a._nexts:
                assert len(branch._prevs) == 1
                assert branch._prevs[0] == a
                assert len(branch._nexts) == 0
        finally:
            task_yaml.unlink()

    def test_scenario_n_1_converge(self):
        """场景3：N-1 汇聚"""
        config = self._build_yaml("""
    - name: A
      type: load
      module: csv
      branches:
        - name: A1
          type: clean
          module: common_filter
        - name: A2
          type: x_transformer
          module: standard_scaler
    - name: B
      type: plot
      module: correlation_heatmap
""")
        task_yaml = Path("tests/test_scenario_3.yaml")
        task_yaml.write_text(config)
        try:
            workflow = Builder(str(task_yaml)).build()
            assert len(workflow.starts) == 1
            a = workflow.starts[0]
            # A2 (x_transformer) 为无子节点的孤儿分支，按规则剪除；只剩 A1
            assert len(a._nexts) == 1
            a1 = a._nexts[0]
            assert a1.conf.name == "A1"

            assert len(a1._nexts) == 1
            b = a1._nexts[0]
            assert b.conf.name == "B"
            assert len(b._prevs) == 1
            assert set([p.conf.name for p in b._prevs]) == {"A1"}
        finally:
            task_yaml.unlink()

    def test_scenario_n_1_converge_placeholder_start(self):
        """场景3b：N-1 汇聚（虚拟节点做起点）"""
        config = self._build_yaml("""
    - branches:
        - name: A1
          type: clean
          module: common_filter
        - name: A2
          type: x_transformer
          module: standard_scaler
    - name: B
      type: plot
      module: correlation_heatmap
""")
        task_yaml = Path("tests/test_scenario_3b.yaml")
        task_yaml.write_text(config)
        try:
            workflow = Builder(str(task_yaml)).build()
            # A2 (x_transformer) 为无子节点的孤儿分支，按规则剪除；placeholder 退化为单分支，A1 直接做起点
            assert len(workflow.starts) == 1
            start_names = {s.conf.name for s in workflow.starts}
            assert start_names == {"A1"}

            b = None
            for start in workflow.starts:
                if start._nexts:
                    b = start._nexts[0]
                    break
            assert b is not None
            assert b.conf.name == "B"
            assert len(b._prevs) == 1
        finally:
            task_yaml.unlink()

    def test_scenario_n_n_complex(self):
        """场景4：N-N 复杂"""
        config = self._build_yaml("""
    - name: A
      type: load
      module: csv
      branches:
        - name: A1
          type: clean
          module: common_filter
        - name: A2
          type: x_transformer
          module: standard_scaler
    - branches:
        - name: B1
          type: plot
          module: correlation_heatmap
        - name: B2
          type: plot
          module: correlation_heatmap
""")
        task_yaml = Path("tests/test_scenario_4.yaml")
        task_yaml.write_text(config)
        try:
            workflow = Builder(str(task_yaml)).build()
            assert len(workflow.starts) >= 1

            all_tasks = set()
            visited = set()

            def collect(task):
                if task in visited:
                    return
                visited.add(task)
                all_tasks.add(task.conf.name)
                for n in task._nexts:
                    collect(n)

            for start in workflow.starts:
                collect(start)

            # A2 (x_transformer) 为孤儿分支被剪除，其余保留
            for name in ("A", "A1", "B1", "B2"):
                assert name in all_tasks
            assert "A2" not in all_tasks
        finally:
            task_yaml.unlink()

    def test_scenario_mixed(self):
        """场景5：混合场景（steps + branches）"""
        config = self._build_yaml("""
    - name: A
      type: load
      module: csv
      steps:
        - name: A1
          type: clean
          module: common_filter
        - name: A2
          type: x_transformer
          module: standard_scaler
      branches:
        - name: A_a
          type: plot
          module: correlation_heatmap
        - name: A_b
          type: plot
          module: correlation_heatmap
    - name: B
      type: model
      module: XGBoost
""")
        task_yaml = Path("tests/test_scenario_5.yaml")
        task_yaml.write_text(config)
        try:
            workflow = Builder(str(task_yaml)).build()
            assert len(workflow.starts) == 1

            a = workflow.starts[0]
            assert len(a._nexts) == 3

            a1 = next(t for t in a._nexts if t.conf.name == "A1")
            assert {t.conf.name for t in a._nexts} == {"A1", "A_a", "A_b"}

            # A2 (x_transformer) 为 A1 后继的无 model 孤儿，按规则剪除；A1 直接挂到 B
            assert len(a1._nexts) == 1
            b = a1._nexts[0]
            assert b.conf.name == "B"
            assert len(b._prevs) == 3
            assert set(p.conf.name for p in b._prevs) == {"A1", "A_a", "A_b"}
        finally:
            task_yaml.unlink()


class TestExecutionOrderLIFO:
    """测试 LIFO 拓扑序调度的执行顺序：深度优先，避免层级长尾"""

    def setup_method(self):
        from mflowy.driver.discover import ensure_discovered

        ensure_discovered()  # 惰性扫描：先补全真实注册表，再覆盖 recording handler
        self.original_registry = _REGISTRY.copy()
        self.order: list[str] = []

        from mflowy.middlewares.mlflow_log import mlflow_log
        from mflowy.middlewares.stop_on_error import stop_on_error

        def recording_chain(ctx):
            """裸桩 + 真实中间件链：Workflow.run 断言每个任务在 mlflow run 内执行（last_active_run 非空）"""

            def record(c):
                self.order.append(c.conf.name)
                return None

            return stop_on_error(ctx, lambda c: mlflow_log(c, record))

        for step_type in StepType:
            for name in ("csv", "common_filter", "standard_scaler", "XGBoost", "correlation_heatmap"):
                _REGISTRY[(step_type, name)] = recording_chain

    def teardown_method(self):
        _REGISTRY.clear()
        _REGISTRY.update(self.original_registry)

    def test_lifo_unblocks_downstream_chain_immediately(self, monkeypatch, tmp_path):
        """LIFO：encoder 完成后 RF 立即执行，不等同层 XGB/LGBM/CAT

        构造 house_prices 缩影：load → cv → [XGB, LGBM, CAT, encoder→RF] → plot
        LIFO 期望：load, cv, encoder, RF, CAT, LGBM, XGB, plot
        - encoder 是 cv 的最后声明 branch，LIFO 让它先 pop
        - RF 解锁后立即压栈顶，下个 pop 即执行
        - 剩余 branch 按逆序：CAT, LGBM, XGB
        """
        from types import SimpleNamespace

        monkeypatch.setattr(Workflow, "_setup_mlflow", lambda self: SimpleNamespace(name="t", experiment_id=""))

        config = """workflow:
  name: test_lifo
  steps:
    - name: load
      type: load
      module: csv
    - name: cv
      type: cross_validate
      module: csv
    - branches:
        - name: XGB
          type: model
          module: XGBoost
        - name: LGBM
          type: model
          module: XGBoost
        - name: CAT
          type: model
          module: XGBoost
        - name: encoder
          type: x_transformer
          module: standard_scaler
          steps:
            - name: RF
              type: model
              module: XGBoost
    - name: plot
      type: plot
      module: correlation_heatmap
"""
        task_yaml = tmp_path / "test_lifo.yaml"
        task_yaml.write_text(config)

        workflow = Builder(str(task_yaml)).build()
        workflow.run()

        assert self.order == ["load", "cv", "encoder", "RF", "CAT", "LGBM", "XGB", "plot"], (
            f"Actual order: {self.order}"
        )
