"""内核默认中间件：handler 装饰器的固定尾链（见 handler.py）。

- mlflow_log: 为每个 handler 自动创建独立 MLflow Run，记录业务语义信息。
  Handler 内部如需嵌套 Run（如 fold/trial），在业务代码中自行用 nested=True 创建。
- stop_on_error: 全局错误处理——stop_on_error=True（默认）异常透传终止工作流；
  False 记录日志返回 None 继续执行。

属于 driver 内核（非插件 SDK）的原因：handler 装饰器在装饰期 import 两者，
且不依赖任何能力族（builtin_plugins）的类型。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

import mlflow
from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils.logging import is_verbose
from mflowy.utils.mlflow import active_experiment_id, workflow_tags

logger = logging.getLogger(__name__)


def _space_to_json(obj: object) -> object:
    """将 ContinuousSpace 实例转为 JSON 可序列化类型（用于 mlflow.log_params）。

    study 延迟导入：optuna 属 [modeling] extra，模块级导入会让缺 extra 的环境
    连 load 插件都无法装饰（拆包时修复的 prior bug）。
    """
    from mflowy.utils.study import ContinuousSpace

    if isinstance(obj, ContinuousSpace):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _flatten_params(params: dict, _prefix: str = "") -> dict:
    """过滤并展平参数，供 mlflow.log_params 使用。

    - 跳过 None 值
    - 标量类型（str/int/float/bool）原样保留
    - dict 递归展平，用 . 连接键名
    - tuple 保持圆括号用 str() 转换
    - 其他复杂类型 json.dumps 序列化（ContinuousSpace 由本模块 _space_to_json 处理）
    """
    flat = {}
    for k, v in params.items():
        key = f"{_prefix}.{k}" if _prefix else k
        if v is None:
            continue
        if isinstance(v, dict):
            flat.update(_flatten_params(v, _prefix=key))
        elif isinstance(v, (str, int, float, bool)):
            flat[key] = v
        elif isinstance(v, tuple):
            flat[key] = str(v)
        else:
            flat[key] = json.dumps(v, ensure_ascii=False, default=_space_to_json)
    return flat


def mlflow_log(ctx: Context, next: Handler):
    """为每个 handler 自动创建独立 MLflow Run，仅记录 MLflow 无法自动获取的业务语义信息。"""
    run_name = ctx.conf.name
    # 显式 experiment_id：mlflow 的 _active_experiment_id 是进程级全局，并发 workflow 会串库
    with mlflow.start_run(run_name=run_name, experiment_id=active_experiment_id()) as run:
        ctx._id = run.info.run_id
        step = f"{ctx.conf.type}.{ctx.conf.module}"
        logger.info(f'{step}:<RunInfo: run-name="{run.info.run_name}", run-id="{run.info.run_id}">')
        mlflow.set_tag("mflowy.step", step)
        if extra := workflow_tags():  # Workflow.run(tags=...) 注入的 run 级指纹（文件哈希等）
            mlflow.set_tags(extra)
        mlflow.log_params(_flatten_params(ctx.conf.params))
        try:
            result = next(ctx)
        except Exception as e:
            mlflow.set_tag("mlflow.note.content", repr(e))
            raise
    print()  # handler之间空一行
    return result


def stop_on_error(task: Context, next: Handler):
    try:
        return next(task)
    except Exception as e:
        if task.conf.stop_on_error:
            logger.error(
                f"Task [{task.conf.type}.{task.conf.module}]{task.conf.name} failed: {e}", exc_info=is_verbose()
            )
            raise
        logger.warning(
            f"Task [{task.conf.type}.{task.conf.module}]{task.conf.name} failed and skipped: {e}", exc_info=is_verbose()
        )
        return e
