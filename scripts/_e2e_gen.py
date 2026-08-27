"""查询最新"糖尿病多模型对比"实验的 experiment-id + XGB run_id。

输出两行，供 Makefile e2e 拼装：
    line 1: experiment-id —— 给 modeling --prune-missing / --experiment-id（resume）
    line 2: XGB=<run_id>   —— 给 shap-explanation --model（强制 module=run_id 格式）

modeling 命令仍接 --experiment-id 走批量自动查询；shap-explanation 自 CLI 重构后
强制显式 run_id，所以脚本侧直接吐 preformatted 字符串，Makefile 不再拼。
"""

import mlflow
from mflowy.utils.mlflow import search_experiment_model_run_ids, setup
from mflowy.utils.path import set_task_dir


def main() -> None:
    set_task_dir("examples/diabetes/diabetes.yaml")
    setup()

    exps = mlflow.search_experiments(
        filter_string="name LIKE '糖尿病多模型对比%'",
        order_by=["creation_time DESC"],
        max_results=1,
    )
    if not exps:
        print("ERROR: 未找到 糖尿病多模型对比 实验", flush=True)
        exit(1)

    exp_id = exps[0].experiment_id
    print(exp_id)

    run_id_map = search_experiment_model_run_ids(exp_id)
    xgb_run_id = run_id_map.get("XGB")
    if not xgb_run_id:
        print(f"ERROR: 实验 {exp_id} 中未找到 model.XGB 的 FINISHED run", flush=True)
        exit(1)
    print(f"XGB={xgb_run_id}")


if __name__ == "__main__":
    main()
