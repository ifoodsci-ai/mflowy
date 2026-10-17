"""model 族步骤改写工厂（复用旧实验）与 x_transformer 结构剪枝规则。

知识归属：rewrite recipe 知道 type=="model" / module=="loader" 的参数契约
（run_id 注入）与 ("loader","predict") 不消费 transformer 的拓扑——
这是 model 族 loader 插件的自描述知识，随词汇主人居住。契约（BuilderOption /
StructuralRule）在 driver；装配（何时 prune/resume）在 mcp。
"""

import logging
from typing import TYPE_CHECKING

from mflowy.driver.builder import BuilderOption, StructuralRule
from mflowy.driver.config import NodeConf, node_key
from mflowy.utils.mlflow import search_experiment_model_run_ids, setup

if TYPE_CHECKING:
    from mflowy.driver.context import Context

logger = logging.getLogger(__name__)


def _parse_model_arg(model: str | None) -> tuple[str, str]:
    """解析 model 参数为 (module, run_id)。

    - None / 空白：自动模式（返回 ("", "")，配合 experiment_id 全量查询）
    - 裸 run_id：经 ``mlflow.get_run`` 读 ``mflowy.node`` tag（node_key 形）解析 module
      （``model__XGB`` → ``XGB``）；非 model 族 run 抛 ValueError

    旧 ``module=run_id`` 与裸 module 名形态已随 flavor 前缀退役——
    边界契约见 mcp job_provider/local.py: ``_validate_model_arg``。
    """
    if not model or not model.strip():
        return "", ""
    rid = model.strip()
    if "=" in rid:
        raise ValueError(f"model 参数必须是裸 run_id（module=run_id 格式已废弃），got: {model!r}")
    setup()  # get_run 前对齐 tracking URI（幂等；与 search_experiment_model_run_ids 同约定）
    import mlflow
    from mflowy.driver.config import split_node_key

    node = mlflow.get_run(rid).data.tags.get("mflowy.node", "")
    ns, module = split_node_key(node) if node else ("", "")
    module = module if ns == "model" else ""
    if not module:
        raise ValueError(f"run {rid} 不是 model 族节点（mflowy.node={node!r}）")
    return module, rid


def _resolve_run_id_map(experiment_id: str | None, module: str, run_id: str) -> dict[str, str]:
    """构造 {module: run_id} 映射。

    - module + run_id 都给（裸 run_id 经 get_run 解析后恒成立）：直接映射，不查实验
    - 自动模式（modeling --prune-missing / resume）：experiment_id 下所有 FINISHED model run
    """
    if module and run_id:
        return {module: run_id}
    return search_experiment_model_run_ids(experiment_id)


def prune_model_step(experiment_id: str | None = None, model: str | None = None) -> BuilderOption:
    """shap-explanation / modeling --prune-missing 用：命中的 model.xxx 替换为 loader，未命中的 enabled=False 剪枝。

    Args:
        experiment_id: MLflow 实验 ID（查询 FINISHED run）
        model: 单一 model 的裸 run_id（shap 模式用；module 名经 run 的
            ``mflowy.node`` tag 解析）。None：自动模式（modeling --prune-missing，
            命中 experiment_id 下所有 model）

    工厂时一次性完成：解析 model 参数 + MLflow 查询，构造 {module: run_id} dict；
    option 闭包内仅做 dict 查表，无 MLflow 调用。

    - 命中：module 改写为 loader，注入 {run_id}
    - 单 module 模式且当前 step 不是目标：enabled=False
    - 查询无 FINISHED run：warning + enabled=False（由 _parse_step_dicts 剪枝）
    - 已是 loader（module == 'loader'）：跳过不动
    """
    target_module, explicit_run_id = _parse_model_arg(model)
    run_id_map = _resolve_run_id_map(experiment_id, target_module, explicit_run_id)

    def option(node: NodeConf) -> NodeConf:
        if node.disabled or node.namespace != "model" or node.node_name == "loader":
            return node
        if target_module and node.node_name != target_module:
            node.disabled = True
            return node
        rid = run_id_map.get(node.node_name)
        if not rid:
            logger.warning(
                "实验 %s 中未找到 %s 的 FINISHED run，跳过该模型",
                experiment_id,
                node.node_key,
            )
            node.disabled = True
            return node
        node.params = {"run_id": rid}
        node.node_key = node_key(node.namespace, "loader")
        return node

    return option


def resume_model_step(experiment_id: str) -> BuilderOption:
    """modeling resume 用：一次性查询实验下所有 FINISHED model run，命中的替换为 loader（恢复），
    未命中的保持原状（继续训练）。

    工厂时完成 MLflow 查询，构造 {module: run_id} dict；
    option 闭包内仅做 dict 查表，无 MLflow 调用。

    - 命中：module 改写为 loader，注入 {run_id}（model_evaluation_plots 等
        后续步骤由模型 handler 的 _pipeline 自动执行）
    - 未命中（无 FINISHED run）：返回原 step 不动，由后续 handler 正常训练
    - 已是 loader（module == 'loader'）：跳过不动
    """
    run_id_map = search_experiment_model_run_ids(experiment_id)

    def option(node: NodeConf) -> NodeConf:
        if node.disabled or node.namespace != "model" or node.node_name == "loader":
            return node
        rid = run_id_map.get(node.node_name)
        if not rid:
            return node
        node.params = {"run_id": rid}
        node.node_key = node_key(node.namespace, "loader")
        return node

    return option


def prune_x_transformer_step(ctx: "Context") -> StructuralRule:
    """x_transformer 无血缘消费者即剪（#59 编译后钩子：ctx.nexts = 幸存者视图）。

    消费者 = 消费 transformer 产出的 model 步（非 loader/predict）——血缘 nexts
    即真实消费关系。返回同一 ctx = 保留、None = 剪枝（沿必选边连带传播）。
    """
    if ctx.conf.namespace != "x_transformer":
        return ctx
    if any(c.conf.namespace == "model" and c.conf.node_name not in ("loader", "predict") for c in ctx.nexts):
        return ctx
    return None
