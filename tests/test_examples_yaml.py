"""示例 YAML 反漂移门（#73 决议，B3 根治机制）：examples/**/*.yaml 全量过
validate_modeling_steps——编译即校验（词表 miss/拓扑序/参数边类型门全 fail-loud）。

B3 根因：BREAKING 改词表后示例面无机器反馈——示例从此作为契约被套件钉住
（照示例复现 = 必须可用）。修复前本用例以存量漂移自证红（见 #73 验收标准 1）。
"""

import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
YAMLS = sorted(EXAMPLES.glob("*/*.yaml"))


@pytest.fixture(scope="module")
def validate():
    from mflowy.model.tools import validate_modeling_steps

    return validate_modeling_steps


@pytest.mark.parametrize("yaml_path", YAMLS, ids=lambda p: p.parent.name)
def test_example_yaml_passes_validate(yaml_path, validate):
    """每个示例 YAML 都是合法 modeling_steps 片段（当前词表下可编译）。"""
    report = validate(yaml_path.read_text(encoding="utf-8"))
    assert report.get("valid"), f"{yaml_path.relative_to(ROOT)}: {report.get('error')}"


def test_examples_discovered():
    """守门本身有效：至少 8 个示例 YAML 在遍历面内（防 glob 失效静默空转）。"""
    assert len(YAMLS) >= 8, f"示例 YAML 遍历数异常: {len(YAMLS)}"
