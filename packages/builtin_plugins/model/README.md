# mflowy-model（建模域）

MFlowy 建模域包：建模训练 MDP 的动作域——x_y / cross_validation / x_transformer / model /
plot(model_evaluation + model_interpretability)。CPU 建模栈为硬依赖；torch 栈在 `[deep]`
extra（uvx 安装需 `--index https://download.pytorch.org/whl/cpu`，仅 MLP flavor 需要——
缺它时 `list-nodes` 降级标注，不影响其余能力）。

- 动作族：`model/`（XGB/LGBM/CAT/RF/MLP/auto/loader/predict/search_input）、
  `x_transformer/`（编码 ×5 + 数值变换 ×8）、`cross_validation.py`（split + SplitSpec 单文件）、
  非归组 `x_y.py`（任务目标划分 + 任务类型推理）
- 共享件：`middlewares/`（log_cv / log_prediction / log_search_input）、`plot/`（评估/解释图，
  主题门面复用 `mflowy.utils.plot`）
- 类型：`model/types.py`（MetricName / ModelPipeline / ModelLoader；TASKTYPE 再导出自 data）
- 标注：`extras.py`（`model/mlp → [deep]`，0.2.0 起唯一可选 extra）
- 判据：镜像路径有对应测试（`tests/`），`uv sync` 后 `list-nodes` 可见（新模块须重跑 `uv sync`）

## 族目录声明义务（.mdp_family——会话 SDK 契约面）

建模会话的族目录 = 运行期 discover() 的 model 词表过滤带 `.mdp_family` 声明者——
**强制完整声明：未声明 = 对会话不可见**（DAG 词表不受影响）。第三方模型作者在节点
函数上声明即入目录（第二种扩展形态的对偶面：加节点 vs 加整个环境）：

```python
MY_MODEL.mdp_family = {
    "resource_axis": "n_estimators",  # 预算维（SH 两段式：搜索期截断、终训升满）
    "bounds": (50, 2000),  # 资源轴边界（开局先验的校验域）
    "partial_default": 150,  # partial 资源缺省（research-review 可覆盖）
    "full_default": 1000,  # 终训资源缺省
    "incremental_eval": True,  # 资源轴可增量评估（boosting staged 预测）——
    # False 则 Hyperband 诚实退位（无剪枝路径）
    "structure": MY_MODEL.structure,  # 相容性自声明（cat_support/scaling_need/min_samples）
}
```

签名即搜索空间的单一真源（默认 `ContinuousSpace`/`DiscreteSpace` = 动态维、常量默认 =
静态固定）——`.mdp_family` 不复制空间（防第二真源漂移）。破坏性变更受 CHANGELOG
语义化版本约束。
