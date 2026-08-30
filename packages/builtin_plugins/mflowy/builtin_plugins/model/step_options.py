"""model 族步骤改写工厂（复用旧实验）与 x_transformer 结构剪枝规则。

知识归属：rewrite recipe 知道 type=="model" / module=="loader" 的参数契约
（flavor/run_id 注入）与 ("loader","predict") 不消费 transformer 的拓扑——
这是 model 族 loader 插件的自描述知识，随词汇主人居住。契约（BuilderOption /
StructuralRule）在 driver；装配（何时 prune/resume）在 mcp。
"""

import logging

from mflowy.driver.builder import BuilderOption
from mflowy.driver.config import StepConf
from mflowy.utils.mlflow import search_experiment_model_run_ids

logger = logging.getLogger(__name__)


def _parse_model_arg(model: str | None) -> tuple[str, str]:
    """解析 model 参数为 (module, run_id)。

    - None 或 ""：自动模式（返回 ("", "")）
    - "XGB"：单 module 自动查询（返回 ("XGB", "")）
    - "XGB=abc123"：显式 run_id（返回 ("XGB", "abc123")）

    自动 strip 空白；module/run_id 任一为空抛 ValueError。
    """
    if not model:
        return "", ""
    if "=" in model:
        module, rid = model.split("=", 1)
        module, rid = module.strip(), rid.strip()
        if not module or not rid:
            raise ValueError(f"无效的 model 参数：{model!r}，module 和 run_id 均不可为空（格式：module=run_id）")
        return module, rid
    module = model.strip()
    if not module:
        raise ValueError("model 参数不可为空字符串")
    return module, ""


def _resolve_run_id_map(experiment_id: str, module: str, run_id: str) -> dict[str, str]:
    """构造 {module: run_id} 映射。

    - module + run_id 都给：直接 {module: run_id}
    - 仅 module：从 experiment_id 查询所有 FINISHED run，过滤出该 module
    - 都空：自动模式，返回 experiment_id 下所有 FINISHED model run
    """
    if module and run_id:
        return {module: run_id}
    all_model_runs = search_experiment_model_run_ids(experiment_id)
    if module:
        return {module: all_model_runs[module]} if module in all_model_runs else {}
    return all_model_runs


def prune_model_step(experiment_id: str | None = None, model: str | None = None) -> BuilderOption:
    """shap-explanation / modeling --prune-missing 用：命中的 model.xxx 替换为 loader，未命中的 enabled=False 剪枝。

    Args:
        experiment_id: MLflow 实验 ID（查询 FINISHED run）
        model: 单一 model 描述（shap 模式用）。支持三种格式：
            - None：自动模式（modeling --prune-missing 用，命中 experiment_id 下所有 model）
            - "XGB"：单 module，自动查询 run_id
            - "XGB=abc123"：显式 run_id

    工厂时一次性完成：解析 model 参数 + MLflow 查询，构造 {module: run_id} dict；
    option 闭包内仅做 dict 查表，无 MLflow 调用。

    - 命中：module 改写为 loader，注入 {flavor, run_id}
    - 单 module 模式且当前 step 不是目标：enabled=False
    - 查询无 FINISHED run：warning + enabled=False（由 _parse_step_dicts 剪枝）
    - 已是 loader（module == 'loader'）：跳过不动
    """
    target_module, explicit_run_id = _parse_model_arg(model)
    run_id_map = _resolve_run_id_map(experiment_id, target_module, explicit_run_id)

    def option(step: StepConf) -> StepConf:
        if not step.enabled or step.type != "model" or step.module == "loader":
            return step
        if target_module and step.module != target_module:
            step.enabled = False
            return step
        rid = run_id_map.get(step.module)
        if not rid:
            logger.warning(
                "实验 %s 中未找到 model.%s 的 FINISHED run，跳过该模型",
                experiment_id,
                step.module,
            )
            step.enabled = False
            return step
        step.params = {"flavor": step.module, "run_id": rid}
        step.module = "loader"
        return step

    return option


def resume_model_step(experiment_id: str) -> BuilderOption:
    """modeling resume 用：一次性查询实验下所有 FINISHED model run，命中的替换为 loader（恢复），
    未命中的保持原状（继续训练）。

    工厂时完成 MLflow 查询，构造 {module: run_id} dict；
    option 闭包内仅做 dict 查表，无 MLflow 调用。

    - 命中：module 改写为 loader，注入 {flavor, run_id}（model_evaluation_plots 等
        后续步骤由模型 handler 的 _pipeline 自动执行）
    - 未命中（无 FINISHED run）：返回原 step 不动，由后续 handler 正常训练
    - 已是 loader（module == 'loader'）：跳过不动
    """
    run_id_map = search_experiment_model_run_ids(experiment_id)

    def option(step: StepConf) -> StepConf:
        if not step.enabled or step.type != "model" or step.module == "loader":
            return step
        rid = run_id_map.get(step.module)
        if not rid:
            return step
        step.params = {"flavor": step.module, "run_id": rid}
        step.module = "loader"
        return step

    return option


def _consumes_transformer(conf: StepConf) -> bool:
    """conf 或其后代中存在消费 transformer 的 model 步（非 loader/predict）。

    递归穿透 placeholder 容器——`[scaler, placeholder(branches: [models])]` 形态下
    消费者藏在容器后代里，浅查会把有消费者的 transformer 误剪（review 修复）。
    """
    if conf.type == "model" and conf.module not in ("loader", "predict"):
        return True
    return any(_consumes_transformer(c) for c in (*conf.steps, *conf.branches))


def prune_x_transformer_step(branches: bool, conf: StepConf, nexts: tuple[StepConf, ...]) -> bool:
    if conf.type != "x_transformer":
        return False

    candidates: tuple[StepConf, ...] = ()
    if branches:
        candidates = (conf.steps[:1] if conf.steps else ()) + conf.branches
    elif nexts:
        candidates = (nexts[0],)

    return not any(_consumes_transformer(c) for c in candidates)
