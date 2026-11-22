"""模块可用性标注：模块路径 → 所属 extra。

model 域四个顶层目录（model / cv / x_transformer / plot）全标 ``[local]``
（#216 裁定 Q6.1：本地建模重栈 CPU 六件 + torch 两件住 model 包唯一 extra，
``[deep]`` 真合并）。**目录全标**：标注 = import 失败时的安装提示，瘦装实可用
节点不触发降级、无害；data 全栈硬依赖零标注。纯 dict 无第三方依赖：运行期由
app 层零成本标注目录与 get_node_info——瘦装环境能看清装哪个 extra 可用。
"""

# [local] extra 安装提示单源（#218 瘦装+local provider 永久降级路径）：tools 会话
# 降级 catch 与 session 计算动作的 ModuleNotFoundError 翻译共用——纯常量零依赖，
# 瘦装环境可 import。
LOCAL_INSTALL_HINT = (
    "本地建模重栈缺依赖——安装：pip install 'mflowy-model[local]'（聚合形态 mflowy[local]）；"
    "瘦装环境的计算动作请走 remote provider（switch_job_provider）"
)

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
