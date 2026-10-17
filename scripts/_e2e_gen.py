"""查询最新"糖尿病多模型对比"实验的 experiment-id + XGB run_id。

输出两行，供 Makefile e2e 拼装：
    line 1: experiment-id —— 给 modeling 工具的 experiment_id / prune_missing 参数（复用已训模型）
    line 2: XGB=<run_id>   —— XGB 用来定位，run_id 给 explanation/predict/inverse 的
                             ``model`` 参数（裸 run_id，flavor 前缀已废弃，见
                             job_provider/local.py: _validate_model_arg）

modeling 工具以 experiment_id 参数复用已训模型；下游三工具自 CLI 重构后
强制显式裸 run_id，所以脚本侧直接吐 preformatted 字符串，Makefile 剥前缀后使用。
"""

import sys

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
        print("ERROR: 未找到 糖尿病多模型对比 实验", file=sys.stderr, flush=True)
        exit(1)

    exp_id = exps[0].experiment_id
    print(exp_id)

    run_id_map = search_experiment_model_run_ids(exp_id)
    xgb_run_id = run_id_map.get("XGB")
    if not xgb_run_id:
        print(f"ERROR: 实验 {exp_id} 中未找到 model.XGB 的 FINISHED run", file=sys.stderr, flush=True)
        exit(1)
    print(f"XGB={xgb_run_id}")


if __name__ == "__main__":
    main()
