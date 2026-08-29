import logging
import os
from contextvars import ContextVar
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from .path import task_dir

logger = logging.getLogger(__name__)

# 当前 workflow 的 experiment_id — ContextVar 随线程/请求隔离。
# mlflow fluent 的 _active_experiment_id 是进程级全局，MCP server 并发分发
# 多 workflow 时会互相覆盖；start_run 必须显式携带本值。
_experiment_id_var: ContextVar[str | None] = ContextVar("mflowy_experiment_id", default=None)
_workflow_tags_var: ContextVar[dict[str, str] | None] = ContextVar("mflowy_workflow_tags", default=None)


def set_workflow_tags(tags: dict[str, str] | None):
    """Workflow.run 期注入 run 级指纹 tags；返回 token 供 reset（并发 workflow 经 ContextVar 隔离）。"""
    return _workflow_tags_var.set(tags)


def reset_workflow_tags(token) -> None:
    _workflow_tags_var.reset(token)


def workflow_tags() -> dict[str, str]:
    return _workflow_tags_var.get() or {}


def set_active_experiment(experiment_id: str) -> None:
    """记录当前 workflow 的 experiment_id（供 start_run 显式指定）。"""
    _experiment_id_var.set(experiment_id)


def active_experiment_id() -> str | None:
    """当前 workflow 的 experiment_id；workflow 上下文外为 None（走 mlflow 默认解析）。"""
    return _experiment_id_var.get()


def default_tracking_uri() -> str:
    """固定默认 tracking 库（~/.mflowy/mlflow.db），与任务目录解耦。

    MCP server 进程中 task_dir 随计算任务漂移（或未设置），默认值若
    挂靠 task_dir，查询工具要么报错、要么命中上一次任务的临时库；
    固定路径保证 workflow 侧（setup）与查询侧（mflowy.mcp.tools）同库。
    """
    d = Path.home() / ".mflowy"
    d.mkdir(exist_ok=True)
    return f"sqlite:///{d / 'mlflow.db'}"


def setup():
    """设置 MLflow tracking URI（幂等）。"""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri())
    mlflow.set_tracking_uri(tracking_uri)


def _artifact_local_path(artifact_file: str) -> str:
    """active run 的 artifact_uri + artifact_file 拼绝对路径（file:// 转 local 路径）。"""
    from pathlib import Path
    from urllib.parse import unquote, urlparse

    run = mlflow.active_run()
    if run is None:
        return artifact_file
    uri = f"{run.info.artifact_uri}/{artifact_file}"
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return str(Path(unquote(parsed.path)))
    return uri


def log_table(df, artifact_file: str):
    """mlflow.log_table + print 绝对路径（进 NodeResult.output）。"""
    mlflow.log_table(df, artifact_file)
    print(f"Saved: {_artifact_local_path(artifact_file)}")


def log_dict(dictionary, artifact_file: str):
    """mlflow.log_dict + print 绝对路径（进 NodeResult.output）。"""
    mlflow.log_dict(dictionary, artifact_file)
    print(f"Saved: {_artifact_local_path(artifact_file)}")


def log_figure(fig, filename: str, dpi: int = 300):
    """mlflow.log_figure + print 绝对路径（进 NodeResult.output）。"""
    fig.subplots_adjust(left=0, right=1)  # 保证标题水平居中对齐
    fig.set_dpi(dpi)
    mlflow.log_figure(
        fig,
        filename,
        save_kwargs={"dpi": dpi, "bbox_inches": "tight"},
    )
    print(f"Saved: {_artifact_local_path(filename)}")


def get_artifact_uri() -> str:
    """返回 MLflow artifact 目录 URI（本地路径），由 _setup_mlflow 调用。"""
    return (task_dir() / ".mlruns").as_uri()


def set_tags(tags: dict):
    mlflow.set_tags(tags)


def append_tag(run_id: str, key: str, value: str):
    """向指定 run 的标签追加值，用逗号分隔。

    用 split(",") 精确集合判断避免子串误匹配（如 "task_2" 在 "task_21" 中）。
    """
    run = mlflow.get_run(run_id)
    current = run.data.tags.get(key, "")
    new_value = _compute_appended_value(current, value)
    if new_value is not None:
        MlflowClient().set_tag(run_id, key, new_value)


def _compute_appended_value(current: str, value: str) -> str | None:
    """纯函数：返回应写入的新值，None 表示无需更新。

    用 split(",") 精确集合判断避免子串误匹配（如 "task_2" 在 "task_21" 中）。
    """
    existing = current.split(".") if current else []
    if value in existing:
        return None
    return f"{current}.{value}" if current else value


def get_tag(run_id: str, key: str) -> str:
    run = mlflow.get_run(run_id)
    return run.data.tags.get(key, "")


def search_experiment_model_run_ids(experiment_id: str) -> dict[str, str]:
    """批量查询实验中所有 ``tags.mflowy.step LIKE 'model.%'`` 的 FINISHED run_id。

    一次 MLflow 查询返回 {module: run_id} 字典，供 model 族 step_options 闭包消费——
    避免 option 逐 step 触发 N 次查询。

    - 实验不存在（或已删除）时 logger.warning 提示并返回空 dict
    - ``mflowy.step`` 取点分后缀作为 module 名（如 ``model.XGB`` → ``XGB``）
    - 仅返回 ``status='FINISHED'`` 的 run，避免拿到 FAILED/KILLED 的废 run_id
    - 同 module 多条 FINISHED run 时取 start_time 最早的一条（order_by ASC + dict 覆盖语义）
    - 包含 ``model.loader``（prune/resume/shap 产生的 loader run）——step_options 内部用
      ``step.module == 'loader'`` 跳过，dict 中 ``loader`` entry 自然不被消费

    Args:
        experiment_id: MLflow 实验 ID。

    Returns:
        {module_name: run_id}，例如 {"XGB": "abc123", "LGBM": "def456", "loader": "xyz"}；
        实验不存在时返回 {}。

    Example:
        >>> run_id_map = search_experiment_model_run_ids("123")
        >>> # run_id_map 供 model/step_options 的 prune_model_step / resume_model_step 工厂消费
    """
    setup()
    try:
        exp = mlflow.get_experiment(experiment_id)
    except MlflowException:
        exp = None
    if exp is None:
        logger.warning("实验`%s`不存在", experiment_id)
        return {}
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mflowy.step LIKE 'model.%' AND attributes.status = 'FINISHED'",
        order_by=["start_time ASC"],
        output_format="list",
    )
    assert isinstance(runs, list)
    result: dict[str, str] = {}
    for r in runs:
        step = r.data.tags.get("mflowy.step", "")
        module_name = step.split(".", 1)[1] if "." in step else step
        result[module_name] = r.info.run_id
    return result
