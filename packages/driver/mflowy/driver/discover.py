"""插件目录与解析。entry point name 即身份：``step.module``。

词表来源（按序合并，后组覆盖前组）：

- ``mflowy.builtin_plugins``：本包内置能力，由 ``hatch_metadata.py`` 构建期扫描 ``mflowy.builtin_plugins.**`` 生成
- ``mflowy.plugins``：第三方插件包，声明即注册（随 uv/pip 安装进入环境）

全部查询（``discover``/``has``/``list_all``）只读元数据零 import；import 惰性发生在
``_load_fn``，且以 ``@handler`` 双属性（.handler/.convert_params）校验声明一致性，坏声明 fail-loud。
"""

import logging
from collections.abc import Callable
from functools import cache
from importlib.metadata import EntryPoint, entry_points

logger = logging.getLogger(__name__)

GROUPS = ("mflowy.builtin_plugins", "mflowy.plugins")  # 后组覆盖前组


@cache
def discover() -> dict[str, dict[str, EntryPoint]]:
    """{step: {module: EntryPoint}}，纯元数据零 import"""
    table: dict[str, dict[str, EntryPoint]] = {}
    for g in GROUPS:
        for ep in entry_points(group=g):
            step, sep, module = ep.name.partition(".")
            if not sep or not step or not module:
                logger.warning(f"entry point 名 {ep.name!r} 不满足 step.module（{g}: {ep.value}），跳过")
                continue
            if prev := table.get(step, {}).get(module):
                level = logger.warning if prev.group == g else logger.info
                level(f"插件 {ep.name} 冲突：{ep.group}:{ep.value} 覆盖 {prev.group}:{prev.value}")
            table.setdefault(step, {})[module] = ep
    return table


@cache
def _load_fn(step: str, module: str) -> Callable | None:
    """解析 entry point 并校验 @handler 双属性标记，坏声明 fail-loud"""
    ep = discover().get(step, {}).get(module)
    if ep is None:
        return None
    fn = ep.load()
    missing = [m for m in ("handler", "convert_params") if not hasattr(fn, m)]
    if missing:
        raise ValueError(f"entry point {ep.group}:{ep.name} = {ep.value} 未标注 @handler（缺少 {missing}）")
    return fn


def get(step: str, module: str) -> Callable:
    """严格版：调度/校验入口。miss → ModuleNotFoundError + available 列表"""
    fn = _load_fn(step, module)
    dispatch = getattr(fn, "handler", None)
    if dispatch is None:
        available = sorted(discover().get(step, {}))
        raise ModuleNotFoundError(f"Module '{module}' not found for step '{step}'. Available: {available}")
    return dispatch


def has(step: str, module: str) -> bool:
    return module in discover().get(step, {})


def list_all() -> dict[str, list[str]]:
    return {step: sorted(modules) for step, modules in discover().items()}


def get_post_init(step: str, module: str) -> Callable | None:
    """容错版：解析期（StepConf.__post_init__）调用，miss 返 None 不抛错"""
    fn = _load_fn(step, module)
    return getattr(fn, "convert_params", None)
