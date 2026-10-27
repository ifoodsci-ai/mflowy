"""workspace 完整性：六 distribution 锁步版本 + namespace 铁律（无 mflowy/__init__.py）"""

import pathlib
import tomllib

import pytest

ROOT = pathlib.Path(__file__).parent.parent
# 名 → 包目录（相对 ROOT）：两域包住 builtin_plugins/ 分组（与 entry points 组 mflowy.builtin_plugins 对齐）
PACKAGES = {
    "utils": "packages/utils",
    "driver": "packages/driver",
    "mcp": "packages/mcp",
    "data": "packages/builtin_plugins/data",
    "model": "packages/builtin_plugins/model",
}


def _version(pyproject: pathlib.Path) -> str:
    with open(pyproject, "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_lockstep_versions():
    """六 distribution 版本锁步（发布脚本 sed 全仓，此断言兜底）"""
    versions = {"mflowy": _version(ROOT / "pyproject.toml")}
    for p, rel in PACKAGES.items():
        versions[f"mflowy-{p.replace('_', '-')}"] = _version(ROOT / rel / "pyproject.toml")
    assert len(set(versions.values())) == 1, f"版本漂移: {versions}"


@pytest.mark.parametrize("pkg", PACKAGES)
def test_namespace_no_init(pkg):
    """PEP 420 铁律：任何成员不得提供 mflowy/__init__.py（出现即破坏 namespace 合成）"""
    assert not (ROOT / PACKAGES[pkg] / "mflowy" / "__init__.py").exists()


def test_member_import_boundaries():
    """包依赖方向铁律（utils ← driver ← data ← model ← mcp 单向）：
    driver/utils 不得 import 任何域包（内核对插件零感知）；data 不得 import model
    （两 MDP 域隔离，共享件住上游 data）。mcp 是编排层、pyproject 硬依赖双域包，
    合法消费其公开清单如 extras.py；entry point group 名字符串不算 import。"""
    import re

    bans = {
        "driver": r"mflowy\.(data|model)",
        "utils": r"mflowy\.(data|model)",
        "data": r"mflowy\.model",
    }
    # mcp：零域硬依赖——模块级（列首）import 域包禁止；函数内惰性导入（缩进）合法
    violators = []
    for pkg, pat_s in bans.items():
        pat = re.compile(r"^\s*(?:from|import)\s+" + pat_s, re.M)
        for py in (ROOT / PACKAGES[pkg] / "mflowy").rglob("*.py"):
            if pat.search(py.read_text(encoding="utf-8")):
                violators.append(str(py.relative_to(ROOT)))
    mcp_pat = re.compile(r"^(?:from mflowy\.(data|model)|import mflowy\.(data|model))", re.M)
    for py in (ROOT / "packages" / "mcp" / "mflowy").rglob("*.py"):
        if mcp_pat.search(py.read_text(encoding="utf-8")):
            violators.append(str(py.relative_to(ROOT)))
    assert not violators, f"反向依赖: {violators}"
    mcp_hard = " ".join(_pkg_meta("packages/mcp/pyproject.toml")["dependencies"])
    assert "mflowy-data" not in mcp_hard and "mflowy-model" not in mcp_hard, "mcp 零域硬依赖（0b 解耦不变量）"


def _pkg_meta(rel: str) -> dict:
    with open(ROOT / rel, "rb") as f:
        return tomllib.load(f)["project"]


def test_mcp_extra_is_optional_not_hard():
    """入口形态分离：SDK/遥测三依赖住 [mcp] extra（server 形态），非硬依赖——
    裸装 mflowy 即可用 runner/cmd，不拖 SDK"""
    meta = _pkg_meta("packages/mcp/pyproject.toml")
    hard = " ".join(meta["dependencies"])
    assert "mcp>=" not in hard and "agentcat" not in hard and "machineid" not in hard
    mcp_extra = meta["optional-dependencies"]["mcp"]
    joined = " ".join(mcp_extra)
    assert "mcp>=2.0.0" in joined and "agentcat" in joined and "py-machineid" in joined


def test_aggregate_extras_are_domain_selectors():
    """聚合包 extras = 域选择器（0.2.0）：data / model / mcp / deep——
    [stats]/[modeling] 退役，装哪个域包即得哪个域（重栈为域包硬依赖）"""
    extras = _pkg_meta("pyproject.toml")["optional-dependencies"]
    assert set(extras) == {"data", "model", "mcp", "deep"}
    assert extras["data"] == ["mflowy-data"]
    assert extras["model"] == ["mflowy-model"]
    assert extras["mcp"] == ["mflowy-mcp[mcp]"]
    assert extras["deep"] == ["mflowy-model[deep]"]
    # 唯一可选 extra 在 model 包；data 全栈硬依赖
    assert set(_pkg_meta("packages/builtin_plugins/data/pyproject.toml").get("optional-dependencies", {})) == set()
    assert set(_pkg_meta("packages/builtin_plugins/model/pyproject.toml")["optional-dependencies"]) == {"deep"}


def test_console_scripts():
    """双入口重命名：恰好两条命令——mflowy=CLI（runner）、mflowy-mcp=MCP server；旧名（cmd/mcpSrv）消亡"""
    scripts = _pkg_meta("packages/mcp/pyproject.toml")["scripts"]
    assert scripts == {
        "mflowy": "mflowy.mcp.runner:main",
        "mflowy-mcp": "mflowy.mcp.server:main",
    }


def test_skill_tool_names_in_tools_list():
    """M11：剧本层（.agents/skills/**/SKILL.md，全 git 追踪）的反引号调用形引用
    不超纲——工具名 ⊆ tools/list（mflowy.tools entry points：平台 + 域静态声明）、
    节点 ⊆ 运行期目录、动作面 ⊆ 八动作面词表。改名漂移（SKILL 引用已退役工具/节点）
    即测试红。"""
    import re
    from importlib.metadata import entry_points

    from mflowy.driver.discover import list_all
    from mflowy.model.session import ACTIONS

    tools_vocab = {ep.name for ep in entry_points(group="mflowy.tools")}
    nodes_vocab = {name for names in list_all().values() for name in names}
    # start(...) = start_session 的剧本简写（clean/model 域 SKILL 惯例）
    allowed = tools_vocab | nodes_vocab | set(ACTIONS) | {"start"}

    skills_dir = ROOT / ".agents" / "skills"
    assert skills_dir.is_dir(), "M11：skills 须全 git 追踪入仓（.agents/skills/ 缺失即断链）"
    used: dict[str, str] = {}
    for skill in skills_dir.rglob("SKILL.md"):
        for tok in re.findall(r"`([a-z][a-z0-9_]*)\(", skill.read_text(encoding="utf-8")):
            used.setdefault(tok, str(skill.relative_to(ROOT)))
    unknown = {t: p for t, p in used.items() if t not in allowed}
    assert not unknown, f"SKILL 引用不在运行期词表（工具名 ⊆ tools/list 断链）: {unknown}"
