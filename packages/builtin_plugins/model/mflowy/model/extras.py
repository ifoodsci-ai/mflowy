"""模块可用性标注：模块路径 → 所属 extra。

0.2.0 起唯一可选 extra = model 包的 ``[deep]``（torch / pytorch-lightning，MLP flavor）；
其余能力均为包硬依赖（data 全栈硬依赖、model 的 CPU 建模栈硬依赖），零降级。
纯 dict 无第三方依赖：运行期由 mcp 层零成本标注目录与 get_node_info——
裸环境装不上时能看清装哪个 extra 可用。
"""

_EXTRA_OF_DIR = {
    "model/mlp": "deep",
}


def extra_of(module_path: str) -> str | None:
    """entry point value 的模块路径（如 ``mflowy.model.model.mlp:MLP``）→ 所属 extra；最长目录前缀匹配，非本包路径返 None。"""
    parts = module_path.split(":")[0].split(".")
    try:
        rest = parts[parts.index("mflowy") + 2 :]  # 去掉 mflowy.model 前缀
    except (ValueError, IndexError):
        return None
    for n in range(len(rest), 0, -1):
        if hit := _EXTRA_OF_DIR.get("/".join(rest[:n])):
            return hit
    return None
