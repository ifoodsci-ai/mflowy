"""workspace 完整性：五 distribution 锁步版本 + namespace 铁律（无 mflowy/__init__.py）"""

import pathlib
import tomllib

import pytest

ROOT = pathlib.Path(__file__).parent.parent
PACKAGES = ["utils", "driver", "builtin_plugins", "mcp"]


def _version(pyproject: pathlib.Path) -> str:
    with open(pyproject, "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_lockstep_versions():
    """五 distribution 版本锁步（发布脚本 sed 全仓，此断言兜底）"""
    versions = {"mflowy": _version(ROOT / "pyproject.toml")}
    for p in PACKAGES:
        versions[f"mflowy-{p.replace('_', '-')}"] = _version(ROOT / "packages" / p / "pyproject.toml")
    assert len(set(versions.values())) == 1, f"版本漂移: {versions}"


@pytest.mark.parametrize("pkg", PACKAGES)
def test_namespace_no_init(pkg):
    """PEP 420 铁律：任何成员不得提供 mflowy/__init__.py（出现即破坏 namespace 合成）"""
    assert not (ROOT / "packages" / pkg / "mflowy" / "__init__.py").exists()


def test_member_import_boundaries():
    """包依赖方向铁律：driver/utils/mcp 不得 import builtin_plugins（插件实现不是 SDK 依赖；
    entry point group 名字符串不算 import）"""
    import re

    violators = []
    pat = re.compile(r"^\s*(?:from mflowy\.builtin_plugins|import mflowy\.builtin_plugins)", re.M)
    for pkg in ("driver", "utils", "mcp"):
        for py in (ROOT / "packages" / pkg / "mflowy").rglob("*.py"):
            if pat.search(py.read_text()):
                violators.append(str(py.relative_to(ROOT)))
    assert not violators, f"反向依赖: {violators}"
