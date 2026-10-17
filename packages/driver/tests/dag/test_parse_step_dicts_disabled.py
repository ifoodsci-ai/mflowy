"""_parse_step_dicts disabled 剪枝边界语义测试"""

from pathlib import Path

from mflowy.driver.builder import Builder
from mflowy.driver.config import StepConf


def _write(tmp_path: Path, content: str) -> str:
    f = tmp_path / "test.yaml"
    f.write_text(content)
    return str(f)


class TestBranchesContext:
    """branches 参数控制 disabled 的剪枝边界"""

    def test_sequential_disabled_breaks(self, tmp_path):
        """串行上下文 disabled → 剪枝当前及后续（数据依赖）"""
        path = _write(
            tmp_path,
            "name: test\nsteps:\n"
            '  - name: "A"\n    type: load\n    module: csv\n'
            '  - name: "B"\n    type: clean\n    module: drop_missing\n'
            '  - name: "C"\n    type: plot\n    module: taylor\n',
        )

        def disable_b(step: StepConf) -> StepConf:
            if step.name == "B":
                step.enabled = False
            return step

        builder = Builder(path, disable_b)
        names = [s.name for s in builder.config.workflow.steps]
        # A 保留，B 剪枝，C 因 B 失败也剪枝
        assert names == ["A"]

    def test_parallel_disabled_continues(self, tmp_path):
        """并行上下文 disabled → 仅剪枝当前分支，兄弟分支继续"""
        path = _write(
            tmp_path,
            "name: test\nsteps:\n"
            '  - name: "root"\n'
            "    branches:\n"
            '      - name: "A"\n        type: load\n        module: csv\n'
            '      - name: "B"\n        type: clean\n        module: drop_missing\n'
            '      - name: "C"\n        type: plot\n        module: taylor\n',
        )

        def disable_b(step: StepConf) -> StepConf:
            if step.name == "B":
                step.enabled = False
            return step

        builder = Builder(path, disable_b)
        placeholder = builder.config.workflow.steps[0]
        names = [b.name for b in placeholder.branches]
        # A、C 保留，B 剪枝
        assert names == ["A", "C"]
