"""工具注册表镜像契约：mflowy.tools 组（mflowy-bo 部分）≡ bo/tools.py 公有函数。

声明面（mflowy-bo pyproject 静态声明的安装态）↔ 函数面（模块反射）双向差集：
写了函数忘声明 / 删改函数没同步声明 / 声明后未 ``uv sync``（安装态脱节），三者皆红。
与 ``test_model_tools_registry.py`` / ``test_platform_tools_registry.py`` 同构（#210 六工具声明面）。
"""

import importlib
import inspect
from importlib.metadata import entry_points


def test_registry_matches_tools_module():
    tools = importlib.import_module("mflowy.bo.tools")
    declared = {
        ep.name: ep for ep in entry_points(group="mflowy.tools") if ep.dist is not None and ep.dist.name == "mflowy-bo"
    }
    public = {
        n
        for n, f in vars(tools).items()
        if not n.startswith("_") and inspect.isfunction(f) and f.__module__ == tools.__name__
    }
    assert set(declared) == public, (
        f"注册表与 tools.py 失同步: 仅注册表={set(declared) - public} 仅模块={public - set(declared)}"
    )
    assert set(declared) == {
        "bo_load_or_init",
        "bo_ask",
        "bo_prune",
        "bo_diversity_check",
        "bo_tell",
        "bo_export_dataset",
    }
    for name, ep in declared.items():
        fn = ep.load()
        assert callable(fn) and getattr(fn, "__name__", "") == name


def test_search_input_entry_point_lives_in_bo():
    """#175 真迁移：search_input 节点声明面在 mflowy-bo（model 不留副本——单源）。"""
    from importlib.metadata import entry_points

    bo_eps = {ep.name for ep in entry_points(group="mflowy.builtin_plugins") if ep.dist and ep.dist.name == "mflowy-bo"}
    model_eps = {
        ep.name for ep in entry_points(group="mflowy.builtin_plugins") if ep.dist and ep.dist.name == "mflowy-model"
    }
    assert "search_input" in bo_eps
    assert "search_input" not in model_eps
    assert "inverse_optimization" not in {
        ep.name for ep in entry_points(group="mflowy.tools")
    }, "inverse_optimization 已随 BO 落地退役（#179 R4）"
