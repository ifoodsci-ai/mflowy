"""内核默认中间件：node 装饰器的固定尾链（见 node.py）。

- mlflow_log: 为每个 node 自动创建独立 MLflow Run，记录业务语义信息；
  血缘 tag（``mflowy.lineage``，``_prevs`` 派生视图）在本 run 收尾 finally 统一落盘。
  Node 内部如需嵌套 Run（如 fold/trial），在业务代码中自行用 nested=True 创建。

属于 driver 内核（非插件 SDK）的原因：node 装饰器在装饰期 import 两者，
且不依赖任何能力族（builtin_plugins）的类型。

运行失败的调度处置（stop_on_error）已随 #59 叙事重构退役：一切失败 = 剪枝
血缘下游闭包、其余路径继续（见 dag.py）。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict

from mflowy.driver.context import Context
from mflowy.driver.node import Chain
from mflowy.utils.mlflow import active_experiment_id, active_tracking, workflow_tags

logger = logging.getLogger(__name__)


def _mlflow_safe_key(key: str) -> str:
    """mlflow 参数键转义（遥测面专用，#151/#152）：mlflow 键名只收字母数字/_-.空格:斜杠，
    字符集外字符 → ``-XX`` 十六进制（ASCII 按码位、非 ASCII 按 UTF-8 字节——
    ``%``→``-25``、``公斤``→``-e5-85-ac-e6-96-a4``；标记 ``-`` 本身在允许集，
    与原生 ``-xx`` 段可能同形——遥测面可读性优先，解码不承诺）。仅转义遥测面键——业务参数
    面（NodeConf.params/plan.yaml）原样，值不受 mlflow 键名约束。"""
    safe = re.compile(r"[A-Za-z0-9 _\-./:]")

    def esc(ch: str) -> str:
        if safe.fullmatch(ch):
            return ch
        if ord(ch) < 0x80:
            return f"-{ord(ch):02x}"
        return "".join(f"-{b:02x}" for b in ch.encode("utf-8"))

    return "".join(esc(ch) for ch in key)


def _flatten_params(params: dict, _prefix: str = "") -> dict:
    """过滤并展平参数，供 mlflow.log_params 使用。

    - 跳过 None 值
    - 标量类型（str/int/float/bool）原样保留
    - dict 递归展平，用 . 连接键名；叶键经 _mlflow_safe_key 转义（mlflow 键名字符集）
    - tuple 保持圆括号用 str() 转换
    - 其他复杂类型 json.dumps 序列化（dataclass 由标准库 asdict 兜底）
    """
    flat = {}
    for k, v in params.items():
        key = f"{_prefix}.{k}" if _prefix else k
        if v is None:
            continue
        if isinstance(v, dict):
            flat.update(_flatten_params(v, _prefix=key))  # 前缀保持原样——转义只在叶键一次
            continue
        if isinstance(v, (str, int, float, bool)):
            leaf: object = v
        elif isinstance(v, tuple):
            leaf = str(v)
        else:
            leaf = json.dumps(v, ensure_ascii=False, default=asdict)
        flat[_mlflow_safe_key(key)] = leaf
    return flat


def mlflow_log(ctx: Context, next: Chain):
    """为每个 node 自动创建独立 MLflow Run，仅记录 MLflow 无法自动获取的业务语义信息。"""
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
        finally:
            # 声明消费集血缘：成败皆落盘——失败 run 的输入证据
            if lineage := ctx.lineage:
                t.set_tag(ctx.LINEAGE_TAG_KEY, lineage)
    logger.debug(
        "(node 间隔空行)"
    )  # 通道纪律（#55）：空行分隔属 CLI 观感——非 CLI 宿主（MCP 进程内调用）不可强制写 stdout
    return result
