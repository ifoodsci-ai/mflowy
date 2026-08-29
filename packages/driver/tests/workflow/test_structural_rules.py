"""driver 契约：Builder(structural_rules=...) 注入点——内核零词汇，规则随词汇主人"""

from mflowy.driver.builder import Builder


def test_structural_rule_drops_step(tmp_path):
    """规则返回 True → 步骤被剪枝（与 disabled 剪枝同语义）"""
    yaml = tmp_path / "w.yaml"
    yaml.write_text(
        """
workflow:
  name: t
  steps:
    - {name: a, type: load, module: csv}
    - {name: b, type: clean, module: drop_missing}
"""
    )

    def drop_clean(branches, conf, nexts):
        return conf.type == "clean"

    wf = Builder(yaml, structural_rules=(drop_clean,)).build(preview="name")
    tasks = wf._collect_tasks()
    assert [t.conf.type for t in tasks] == ["load"]


def test_no_rules_keeps_all(tmp_path):
    yaml = tmp_path / "w.yaml"
    yaml.write_text(
        """
workflow:
  name: t
  steps:
    - {name: a, type: load, module: csv}
    - {name: b, type: clean, module: drop_missing}
"""
    )
    wf = Builder(yaml).build(preview="name")
    assert len(wf._collect_tasks()) == 2
