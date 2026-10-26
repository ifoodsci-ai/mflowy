"""内核默认中间件：handler 装饰器的固定尾链（见 handler.py）。

- mlflow_log: 为每个 handler 自动创建独立 MLflow Run，记录业务语义信息；
  血缘 tag（``mflowy.lineage``，``_prevs`` 派生视图）在本 run 收尾 finally 统一落盘。
  Handler 内部如需嵌套 Run（如 fold/trial），在业务代码中自行用 nested=True 创建。

属于 driver 内核（非插件 SDK）的原因：handler 装饰器在装饰期 import 两者，
且不依赖任何能力族（builtin_plugins）的类型。

运行失败的调度处置（stop_on_error）已随 #59 叙事重构退役：一切失败 = 剪枝
血缘下游闭包、其余路径继续（见 dag.py）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils.mlflow import active_experiment_id, active_tracking, workflow_tags

logger = logging.getLogger(__name__)


def _flatten_params(params: dict, _prefix: str = "") -> dict:
    """过滤并展平参数，供 mlflow.log_params 使用。

    - 跳过 None 值
    - 标量类型（str/int/float/bool）原样保留
    - dict 递归展平，用 . 连接键名
    - tuple 保持圆括号用 str() 转换
    - 其他复杂类型 json.dumps 序列化（dataclass 由标准库 asdict 兜底）
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
            flat[key] = json.dumps(v, ensure_ascii=False, default=asdict)
    return flat


def mlflow_log(ctx: Context, next: Handler):
    """为每个 handler 自动创建独立 MLflow Run，仅记录 MLflow 无法自动获取的业务语义信息。"""
    t = active_tracking()
    run_name = ctx.conf.name
    # 显式 experiment_id：mlflow 的 _active_experiment_id 是进程级全局，并发 workflow 会串库
    with t.node_run(run_name, active_experiment_id()) as run:
        ctx.adopt_run(run.info.run_id)  # 执行记账上报（身份字段归 Context 所有）
        node = ctx.conf.node_key
        logger.info(f'{node}:<RunInfo: run-name="{run.info.run_name}", run-id="{run.info.run_id}">')
        t.set_tag("mflowy.node", node)
        if extra := workflow_tags():  # DAG.run(tags=...) 注入的 run 级指纹（文件哈希等）
            t.set_tags(extra)
        t.log_params(_flatten_params(ctx.conf.params))
        try:
            result = next(ctx)
        except Exception as e:
            t.set_tag("mlflow.note.content", repr(e))
            raise
        finally:
            # 声明消费集血缘：成败皆落盘——失败 run 的输入证据
            if lineage := ctx.lineage:
                t.set_tag(ctx.LINEAGE_TAG_KEY, lineage)
    logger.debug(
        "(handler 间隔空行)"
    )  # 通道纪律（#55）：空行分隔属 CLI 观感——非 CLI 宿主（MCP 进程内调用）不可强制写 stdout
    return result
