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

    # 同文件幂等：按剥掉 :target 的路径部比对全部 *_file 键——run 级 tag（predict 传
    # x.py:load_X）与 loader 运行期（x.py）不得因引用形式不同而判异
    def _path_of(ref: str) -> str:
        return ref.split(":", 1)[0]

    new_path = _path_of(tags["mflowy.data_file"])
    if any(k.startswith("mflowy.data_file") and _path_of(v) == new_path for k, v in workflow_tags().items()):
        return

    current = workflow_tags()
    n = 2
    while f"mflowy.data_sha256_{n}" in current:
        n += 1
    suffix = "" if "mflowy.data_sha256" not in current else f"_{n}"
    set_workflow_tags(
        {
            **current,
            f"mflowy.data_sha256{suffix}": tags["mflowy.data_sha256"],
            f"mflowy.data_file{suffix}": tags["mflowy.data_file"],
        }
    )
