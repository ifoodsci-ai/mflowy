# mflowy-model（建模域）

MFlowy 建模域包：建模训练 MDP 的动作域——x_y / cv / x_transformer / model /
plot(model_evaluation + model_interpretability)。CPU 建模栈为硬依赖；torch 栈在 `[deep]`
extra（uvx 安装需 `--index https://download.pytorch.org/whl/cpu`，仅 MLP flavor 需要——
缺它时 `list-nodes` 降级标注，不影响其余能力）。

- 动作族：`model/`（XGB/LGBM/CAT/RF/MLP/loader/predict/search_input）、
  `x_transformer/`（编码 ×5 + 数值变换 ×8）、`cross_validation.py`（SplitSpec 契约模块——节点住 cv/ 族目录）、`cv/`（split 参数构建 + load 引用加载——构建与引用分家）、
  非归组 `x_y.py`（任务目标划分 + 任务类型推理）
- 共享件：`middlewares/`（log_cv / log_prediction / log_search_input）、`plot/`（评估/解释图，
  主题门面复用 `mflowy.utils.plot`）
- 类型：`model/types.py`（MetricName / ModelPipeline / ModelLoader；TASKTYPE 再导出自 data）
- 标注：`extras.py`（`model/mlp → [deep]`，0.2.0 起唯一可选 extra）
- 判据：镜像路径有对应测试（`tests/`），`uv sync` 后 `list-nodes` 可见（新模块须重跑 `uv sync`）

## 族节点两行为派生（#62：extract_search_spaces 空否即判据）

- 参数含搜索空间 → **hyperparameters_search 行为**：inner CV 选参（θ*/u_inner 证据），
  不终训不碰 holdout——配 `hyperparameters_search` compute 工具做候选扫描
- 全 pin（空间空）→ **training 行为**：全量拟合 + holdout 终评 + loader
  （`modeling` 工具 / plan 重放 / resume 的生产路径）
- 共享底座 = `_pipeline.hyperparameters_search` + `_pipeline.training`（estimator 机器
  由 `model/utils.make_machinery` 工厂从签名派生——缺省超参单一真源 = 签名）

## 族目录声明义务（.mdp_family——会话 SDK 契约面）

建模会话的族目录 = 运行期 discover() 的 model 词表过滤带 `.mdp_family` 声明者——
**强制完整声明：未声明 = 对会话不可见**（DAG 词表不受影响）。第三方模型作者在节点
函数上声明即入目录（第二种扩展形态的对偶面：加节点 vs 加整个环境）：

```python
MY_MODEL.mdp_family = {
    "structure": MY_MODEL.structure,  # 相容性自声明（cat_support/scaling_need/min_samples）
    "machinery": make_machinery(MY_MODEL, _model_cls),  # 会话直调的 estimator 机器
    # （task, x_preprocessors, seed=None, **overrides) -> (make_pipe, base_params, param_space)
}
MY_MODEL.regularizers = ("my_l2",)  # 正则面自声明（#38 出题规则的正则收窄目标）
```

签名即搜索空间的单一真源（默认 `ContinuousSpace`/`DiscreteSpace` = 动态维、常量默认 =
静态固定）——`.mdp_family` 不复制空间（防第二真源漂移）；`machinery` 经工厂从签名
派生（不手写第二份缺省表）。破坏性变更受 CHANGELOG 语义化版本约束。

## 会话观察面（#35/#64）

- `diagnose.py`：OOF 观察面上的四算子（error_concentrated / slice_collapse / fold_std /
  complexity_edge）——run_hpo 后自动接线（findings 落账本 + 响应浮出），零步数
- `report.py`：OOF 图集（挂 run 下，run_id 寻址）/ 泰勒聚合（终训节点副产物
  fan-in——`taylor_byproducts` 边函数断言各模型 y_test 逐位同一）/ 报告素材
  （`modeling_report_skeleton`：表格从账本渲染 + 图片清单 + 叙事槽）；终局报告不在
  引擎——π 撰写后 `finish(pdf)` 收编进 deliverables（tmp/rename 原子落盘，不编译不验内容）
- `propose_rules.py`：出题规则（rule = 节点附加属性，与 clean 域 `.repair` 同构）——
  findings 驱动层经 `driver.rule.fire/prioritize` 进菜单
