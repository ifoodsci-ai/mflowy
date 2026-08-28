"""load 族共享工具（非插件：hatch 扫描按 ``_EXCLUDED`` 跳过 utils 词干）。"""

from mflowy.utils.file import fingerprint_tags
from mflowy.utils.mlflow import set_workflow_tags, workflow_tags


def set_data_fingerprint(source: str) -> None:
    """load 步数据文件指纹并入 workflow tags。

    由各文件型 loader 在解析出**绝对路径**后显式调用（``ensure_relative_path_under_task_dir``
    语义在调用方保证）：后续节点 run 经 mlflow_log 自动携带；本节点 run 由
    ``log_load_data_fingerprint`` 中间件事后补写。多 load 步：首文件平键
    （``mflowy.data_sha256``），不同文件第 n 个起 ``_n`` 后缀，同文件幂等。
    """
    tags = fingerprint_tags("data", source)
    if not tags:
        return  # 远程引用/本地缺失：指纹只对可核验的本地工件负责

    current = workflow_tags()
    if "mflowy.data_sha256" not in current:
        merged = {**current, **tags}
    elif current.get("mflowy.data_file") == tags["mflowy.data_file"]:
        return  # 同文件幂等
    else:
        n = 2
        while f"mflowy.data_sha256_{n}" in current:
            n += 1
        merged = {
            **current,
            f"mflowy.data_sha256_{n}": tags["mflowy.data_sha256"],
            f"mflowy.data_file_{n}": tags["mflowy.data_file"],
        }
    set_workflow_tags(merged)
