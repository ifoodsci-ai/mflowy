"""通用模型加载 handler。

产物契约：训练步在 parent run 上以 name="final" 持久化唯一 ModelPipeline
（mlflow.sklearn）；本 handler按 source_run_id 取回，惰性恢复为 ModelLoader 门面。
旧 fold 产物（v0.1.x 的 ``{flavor}_{i}`` 命名模型）已废弃——未命中即明确报错指引重训。
"""

from __future__ import annotations

from typing import Annotated

from mflowy.driver.handler import handler

from .types import ModelLoader


@handler(namespace="model")
def loader(
    run_id: Annotated[str, "训练 model 步骤所在 Run 的 id，e.g. <RunInfo: ..., run-id=xxx>"],
):
    """model.loader：从指定 run 恢复唯一最终模型（name='final' 的 ModelPipeline），返回 ModelLoader 门面。

    五 flavor 共用同一路径（产物格式单一：恒 sklearn Pipeline，分界是 pipeline 组成而非 flavor）；
    身份取自 run 的 ``mflowy.step`` tag（如 ``model.XGB`` → "XGB"）。模型按需惰性加载。
    """
    import mlflow

    run = mlflow.get_run(run_id)
    models = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"name = 'final' AND source_run_id = '{run_id}'",
        output_format="list",
    )
    if not models:
        step = run.data.tags.get("mflowy.step", "model.?")
        raise ValueError(
            f"run {run_id} 没有 name='final' 的模型产物（该 run 的 step 为 {step}）。"
            "旧 fold 产物（{flavor}_{i} 命名模型）已随 v0.2 契约废弃——请用原 YAML 重训后再引用。"
        )
    name = run.data.tags.get("mflowy.step", "model").removeprefix("model.") or "model"
    return ModelLoader(name=name, _model_uri=models[0].model_uri)
