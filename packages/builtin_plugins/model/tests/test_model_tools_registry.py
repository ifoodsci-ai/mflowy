"""工具注册表镜像契约：mflowy.tools 组（mflowy-model 部分）≡ tools.py 公有函数。"""

import importlib
import inspect
from importlib.metadata import entry_points


def test_registry_matches_tools_module():
    tools = importlib.import_module("mflowy.model.tools")
    declared = {
        ep.name: ep
        for ep in entry_points(group="mflowy.tools")
        if ep.dist is not None and ep.dist.name == "mflowy-model"
    }
    public = {
        n
        for n, f in vars(tools).items()
        if not n.startswith("_") and inspect.isfunction(f) and f.__module__ == tools.__name__
    }
    assert set(declared) == public, (
        f"注册表与 tools.py 失同步: 仅注册表={set(declared) - public} 仅模块={public - set(declared)}"
    )
    for name, ep in declared.items():
        fn = ep.load()
        assert callable(fn) and getattr(fn, "__name__", "") == name
