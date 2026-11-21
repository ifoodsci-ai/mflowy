"""workspace 完整性：七 distribution 锁步版本 + namespace 铁律（无 mflowy/__init__.py）"""

import pathlib
import tomllib

import pytest

ROOT = pathlib.Path(__file__).parent.parent
# 名 → 包目录（相对 ROOT）：三域包住 builtin_plugins/ 分组（与 entry points 组 mflowy.builtin_plugins 对齐）
PACKAGES = {
    "utils": "packages/utils",
    "driver": "packages/driver",
    "app": "packages/app",
    "data": "packages/builtin_plugins/data",
    "model": "packages/builtin_plugins/model",
    "bo": "packages/builtin_plugins/bo",
}


def _version(pyproject: pathlib.Path) -> str:
    with open(pyproject, "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_lockstep_versions():
    """七 distribution 版本锁步（发布脚本 sed 全仓，此断言兜底）"""
    versions = {"mflowy": _version(ROOT / "pyproject.toml")}
    for p, rel in PACKAGES.items():
        versions[f"mflowy-{p.replace('_', '-')}"] = _version(ROOT / rel / "pyproject.toml")
    assert len(set(versions.values())) == 1, f"版本漂移: {versions}"


@pytest.mark.parametrize("pkg", PACKAGES)
def test_namespace_no_init(pkg):
    """PEP 420 铁律：任何成员不得提供 mflowy/__init__.py（出现即破坏 namespace 合成）"""
    assert not (ROOT / PACKAGES[pkg] / "mflowy" / "__init__.py").exists()


def test_member_import_boundaries():
    """包依赖方向铁律（utils ← driver ← data ← model ← bo ← app 单向，#175 Q2/#243）：
    driver/utils 不得 import 任何域包（内核对插件零感知）；data 不得 import model
    （两 POMDP 域隔离，共享件住上游 data）；bo 不得 import app（app 是编排层，
    pyproject 硬依赖三域包——#243 单一安装形态，模块级 import 域包合法）；
    entry point group 名字符串不算 import。"""
    import re

    bans = {
        "driver": r"mflowy\.(data|model|bo|app)",
        "utils": r"mflowy\.(data|model|bo|app)",
        "data": r"mflowy\.(model|bo|app)",
        "bo": r"mflowy\.app",
    }
    violators = []
    for pkg, pat_s in bans.items():
        pat = re.compile(r"^\s*(?:from|import)\s+" + pat_s, re.M)
        for py in (ROOT / PACKAGES[pkg] / "mflowy").rglob("*.py"):
            if pat.search(py.read_text(encoding="utf-8")):
                violators.append(str(py.relative_to(ROOT)))
    assert not violators, f"反向依赖: {violators}"


def _pkg_meta(rel: str) -> dict:
    with open(ROOT / rel, "rb") as f:
        return tomllib.load(f)["project"]


def test_app_single_install_form():
    """单一安装形态（#243）：mflowy-app = driver + utils + 三域包 + py-machineid——
    零 extras、零 MCP SDK、零 agentcat（CLI 即全词表）"""
    meta = _pkg_meta("packages/app/pyproject.toml")
    hard = " ".join(meta["dependencies"])
    assert "mcp>=" not in hard and "agentcat" not in hard
    for dep in ("mflowy-driver", "mflowy-utils", "mflowy-data", "mflowy-model", "mflowy-bo", "py-machineid"):
        assert dep in hard, f"mflowy-app 缺基础依赖 {dep}"
    assert not meta.get("optional-dependencies"), "mflowy-app 不设 extras（单一安装形态）"


def test_aggregate_single_extra_local():
    """聚合包 extras（#243 单一安装形态）：唯一 extra = [local]（本地建模重栈，
    model 包 [local]，[deep] 真合并）——[mcp] 已随 MCP server 退役删除、不设
    [app] extra（CLI 即全词表，mflowy-app 硬带三域包）"""
    meta = _pkg_meta("pyproject.toml")
    assert set(meta["optional-dependencies"]) == {"local"}
    assert meta["optional-dependencies"]["local"] == ["mflowy-model[local]"]
    assert "mflowy-app" in meta["dependencies"] and "mflowy-mcp" not in " ".join(meta["dependencies"])
    # 唯一可选 extra 在 model 包（= local）；data/bo 零 extras（#184）
    assert set(_pkg_meta("packages/builtin_plugins/data/pyproject.toml").get("optional-dependencies", {})) == set()
    assert set(_pkg_meta("packages/builtin_plugins/bo/pyproject.toml").get("optional-dependencies", {})) == set()
    assert set(_pkg_meta("packages/builtin_plugins/model/pyproject.toml")["optional-dependencies"]) == {"local"}


def test_console_scripts():
    """入口面（#243）：mflowy=CLI（旗标编译面）、mflowy-job=远程任务容器入口
    （#158 wrap 提交面）；mflowy-mcp 已随 MCP server 退役删除"""
    scripts = _pkg_meta("packages/app/pyproject.toml")["scripts"]
    assert scripts == {
        "mflowy": "mflowy.app.cli:main",
        "mflowy-job": "mflowy.app.jobwrap:main",
    }


def _skill_files() -> list[pathlib.Path]:
    """剧本层 SKILL.md 全集（M11/#233 共用：须全 git 追踪入仓）。"""
    skills_dir = ROOT / ".agents" / "skills"
    assert skills_dir.is_dir(), "M11：skills 须全 git 追踪入仓（.agents/skills/ 缺失即断链）"
    return sorted(skills_dir.rglob("SKILL.md"))


def _call_forms(text: str, suffix: str = "") -> set[str]:
    """SKILL 文本里的调用形函数名（两曲面：反引号内联 + 围栏代码块行首），
    可选按后缀过滤（#233 开局权断言用 ``_load_or_start``）。"""
    import re

    called = {t for t in re.findall(r"`([a-z][a-z0-9_]*)\(", text) if t.endswith(suffix)}
    for fence in re.findall(r"^```.*?^```", text, re.MULTILINE | re.DOTALL):
        called |= {t for t in re.findall(r"^([a-z][a-z0-9_]*)\(", fence, re.MULTILINE) if t.endswith(suffix)}
    return called


def test_skill_tool_names_in_tools_list():
    """M11：剧本层（.agents/skills/**/SKILL.md，全 git 追踪）的反引号调用形引用
    不超纲——工具名 ⊆ tools/list（mflowy.tools entry points：平台 + 域静态声明）、
    节点 ⊆ 运行期目录、动作面 ⊆ 八动作面词表。改名漂移（SKILL 引用已退役工具/节点）
    即测试红。"""
    from importlib.metadata import entry_points

    from mflowy.driver.discover import list_all
    from mflowy.model.session import ACTIONS

    tools_vocab = {ep.name for ep in entry_points(group="mflowy.tools")}
    nodes_vocab = {name for names in list_all().values() for name in names}
    # start(...) = start_session 的剧本简写（clean/model 域 SKILL 惯例）
    allowed = tools_vocab | nodes_vocab | set(ACTIONS) | {"start"}

    used: dict[str, str] = {}
    for skill in _skill_files():
        for tok in _call_forms(skill.read_text(encoding="utf-8")):
            used.setdefault(tok, str(skill.relative_to(ROOT)))
    unknown = {t: p for t, p in used.items() if t not in allowed}
    assert not unknown, f"SKILL 引用不在运行期词表（工具名 ⊆ mflowy --help 断链）: {unknown}"


def test_skill_topology_opening_ownership():
    """#233 拓扑契约：开局权归主剧本——`<biz>_load_or_start` 调用形（反引号+调用括号）
    只出现在主技能；prior-survey 家族产 JSON/卡交棒不开局。技能集镜像终名表
    （目录名 = 前言 name 字段，改名回潮/半改名即红）。"""
    import re

    names = {p.parent.name for p in _skill_files()}
    assert names == {
        "data-prior-survey", "model-prior-survey", "bo-prior-survey",
        "model-research-review", "data-clean", "modeling", "bo-suggest",
        "model-predict", "model-explain",
    }, f"#233 技能终名表漂移: {sorted(names)}"

    owners = {
        "data-clean": "data_load_or_start",
        "modeling": "modeling_load_or_start",
        "bo-suggest": "bo_load_or_start",
    }
    for skill in _skill_files():
        text = skill.read_text(encoding="utf-8")
        m = re.search(r"^name:\s*(\S+)", text, re.MULTILINE)
        assert m and m.group(1) == skill.parent.name, f"SKILL 前言 name 与目录名不一致: {skill.parent.name}"
        called = _call_forms(text, suffix="_load_or_start")
        biz = skill.parent.name
        if biz in owners:
            assert called == {owners[biz]}, f"{biz} 开局调用面漂移（须恰持 {owners[biz]}）: {called}"
        else:
            assert not called, f"{biz} 不开局（开局权归主剧本，#233）: {called}"


def test_skill_doc_references_stay_in_repo():
    """#65/M11：剧本参考节引用的仓内路径必须真实入仓（引用被 ignore 的文档 = 断链——
    干净检出里不存在即测试红）。实现路径引用已被剧本层谓词禁止（AGENTS.md 防漂移），
    本测兜「入仓文档」那半边。"""
    import re

    skills_dir = ROOT / ".agents" / "skills"
    assert skills_dir.is_dir()
    broken: dict[str, str] = {}
    for md in skills_dir.rglob("*.md"):
        for ref in re.findall(r"docs/[A-Za-z0-9_./-]+?\.md", md.read_text(encoding="utf-8")):
            if not (ROOT / ref).is_file():
                broken.setdefault(ref, str(md.relative_to(ROOT)))
    assert not broken, f"SKILL 引用了未入仓的文档（M11 引用文档入仓断链）: {broken}"


def test_active_docs_use_pomdp_not_mdp():
    """术语统一（0.2.0 MDP→POMDP）防回潮：活性治理文档零裸 "MDP" 残留。

    排除面各有其据：CHANGELOG（已发布历史条目——追加式纪律）、docs/research（历史
    设计快照，文件名即 mdp）、benchmark_readiness_report（自declared 一次性评估
    快照）、代码文件（.mdp_family 为代码锚点，标识符不随术语改名——
    Handle 已随 #89 改名 POMDPSessionHandle）；CONTEXT.md 的 _Avoid_ 行例外——禁用别名表本就该点名旧称。"""
    import re

    files = [
        ROOT / "AGENTS.md",
        ROOT / "CONTEXT.md",
        ROOT / "README.md",
        *sorted((ROOT / "packages").rglob("README.md")),  # 含 builtin_plugins/{data,model} 两级深
        *sorted((ROOT / "docs").glob("*.md")),
    ]
    files += sorted((ROOT / ".agents" / "skills").rglob("*.md"))

    for f in files:
        text = f.read_text(encoding="utf-8")
        if f.name == "CONTEXT.md":
            text = "\n".join(line for line in text.splitlines() if "_Avoid_" not in line)
        leftover = [line.strip() for line in text.splitlines() if re.search(r"\bMDP\b", line)]
        assert not leftover, f"{f.relative_to(ROOT)} 仍有裸 MDP（术语统一回潮）: {leftover[:3]}"


def test_domain_sessions_implement_handle_protocol():
    """#89 结构等价钉测（#226 ②⑨ 随三切片收敛，#236）：两域 = 会话实体
    （CleaningSession/ModelingSession——⑨ 合规修复，模块级薄门面退役）；响应
    类型 = 契约模型；report_findings 进 ACTIONS（④ 九动作面收编）。"""
    import inspect

    from mflowy.data.session import CleaningSession
    from mflowy.driver.pomdp import Observation, POMDPSessionHandle
    from mflowy.model.session import ACTIONS, ModelingSession

    # 两域实体面：六操作齐备 + typed 形状
    for entity in (CleaningSession, ModelingSession):
        for op in ("load_or_start", "observe", "apply", "rollback", "finish", "abandon"):
            assert hasattr(entity, op), (entity.__name__, op)
        assert list(inspect.signature(entity.load_or_start).parameters)[:2] == ["self", "observation_json"]
    assert list(inspect.signature(CleaningSession.observe).parameters)[:3] == [
        "self",
        "observation_json",
        "prior_updates",
    ]
    assert "report_findings" in ACTIONS  # #226 ④/#236：九动作面收编（工具面薄别名）
    assert (
        "Observation" in inspect.signature(ModelingSession.load_or_start).return_annotation
    )  # (observation_json, o) 二联

    # 响应注解 = 契约模型（六操作面向 π 的三类返回面）
    assert "Observation" in inspect.signature(ModelingSession.apply).return_annotation
    assert "FinishVerdict" in inspect.signature(ModelingSession.finish).return_annotation
    # 协议面自检：六操作齐备（防 Handle 面漂移；start→load_or_start 随 #226 ②）
    for op in ("load_or_start", "observe", "apply", "rollback", "finish", "abandon"):
        assert hasattr(POMDPSessionHandle, op), op
    assert "Observation" in inspect.signature(CleaningSession.observe).return_annotation
    assert Observation.model_config.get("extra") == "forbid"  # 封闭契约
