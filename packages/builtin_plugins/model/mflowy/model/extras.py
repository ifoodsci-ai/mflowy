"""模块可用性标注：模块路径 → 所属 extra。

model 域四个顶层目录（model / cv / x_transformer / plot）全标 ``[local]``
（#216 裁定 Q6.1：本地建模重栈 CPU 六件 + torch 两件住 model 包唯一 extra，
``[deep]`` 真合并）。**目录全标**：标注 = import 失败时的安装提示，瘦装实可用
节点不触发降级、无害；data 全栈硬依赖零标注。纯 dict 无第三方依赖：运行期由
mcp 层零成本标注目录与 get_node_info——瘦装环境能看清装哪个 extra 可用。
"""

_EXTRA_OF_DIR = {
    "model": "local",
    "cv": "local",
    "x_transformer": "local",
    "plot": "local",
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
