# mflowy-builtin-plugins — 内置能力目录

MFlowy 全部内置 ML 能力，同时是**第三方 `mflowy.plugins` 插件包的活参考实现**——本包与第三方插件走完全相同的注册机制（`@handler` 标记 + entry points 声明），抄走本包的 `pyproject.toml` 与 `hatch_metadata.py`、把 entry points 组改成 `mflowy.plugins` 即成一个插件包。

> 目录是快照，实时以 MCP `list_modules` / `get_module_info` 为准（MCP schema 即 API 文档，参数细节不在此复述）。

## 能力目录（step 族 × 模块）

| step | 数据契约（产出） | 模块 |
|------|----------------|------|
| `load` | → 原始 `pd.DataFrame` | `csv` `excel` `parquet` `http` `file` `python`（脚本沙箱经安全扫描） |
| `clean` | `df → df` | 缺失：`drop_missing` `fill_missing`；异常值：`zscore_detector` `iqr_detector`；过滤：`variance_filter` `correlation_filter` `common_filter`；单位：`strip_units`；自定义：`python` |
| `X_y` | → `(X, y, TASKTYPE)` | `x_y`（targets 声明 + 任务类型自动推断） |
| `x_transformer` | → sklearn transformer（每 fold 独立 fit 防泄露） | 类别编码：`onehot_encoder` `ordinal_encoder` `label_encoder` `target_encoder` `hash_encoder`；数值：`standard_scaler` `minmax_scaler` `robust_scaler` `log_transformer` `power_transformer` `numerical_binner` `pca_reducer` `interaction_creator` |
| `cross_validate` | → folds 迭代器（train/val/test 索引） | `simple_cv` `k_fold` `stratified_k_fold` `group_k_fold` `stratified_group_k_fold` `repeated_k_fold` `repeated_stratified_k_fold` `leave_one_out` `leave_one_group_out` |
| `model` | → `ModelLoader`（逐 fold 模型实例） | 训练：`XGB` `LGBM` `CAT` `RF` `MLP`（回归/分类/多分类）；复用与消费：`loader`（按 run_id 载入旧模型）`predict` `search_input`（输入寻优） |
| `plot` | → yield `(df, fig)` | 数据分析：`correlation_heatmap` `numeric_quality_kde_hist` `numeric_scale_box` `target_trend_by_numeric` `target_effect_by_category` `target_association_by_category` `target_separation_by_numeric`；模型评估：`taylor_diagram` `prediction_scatter` `confusion_matrix`；可解释性：`shap_summary` `shap_dependence` `sample_waterfall` |
| `statistic` | → schema DataFrame | `profile`（数据画像）`effect_size`（效应量） |

依赖分层（`extras.py` 目录级标注，构建期守卫全覆盖）：load/clean/X_y/statistic 与 data_analysis/model_evaluation 图表属 `[stats]` extra；x_transformer/cross_validate/model 与 SHAP 图表属 `[modeling]` extra（含 stats）。缺 extra 的环境模块照常出现在目录中——`list_modules` 带 `requires` 标注、`get_module_info` 返回 `available=false` + 所需 extra，而非裸 ImportError。

## 注入器契约（middlewares/）

`middlewares/` 是插件间数据面契约的事实文档——每个 step 族的产出如何被下游消费：

- `getters.py`：`Get*` 按族查询上游 Context 取数据（`GetDF` `GetXy` `GetModel` `GetDatasetLoader`…），可被任意插件直呼
- `inject.py`：`inject_*` 装饰器注入中间件（`inject_df` `inject_X_y` `inject_dataset_loader` `inject_plot_data` 工厂…），handler 函数不直接接触 ctx
- `log_*.py`：领域观测（load 画像 / plot 渲染落盘 / CV 折划分 / df 差异…）
- `df_columns.py`：df 列诊断工具（plot 族共用的列校验）

内核默认尾链（mlflow_log / stop_on_error）在 [mflowy-driver](../driver/README.md) 的 `builtin_middleware.py`，不随插件演化。

## 写一个第三方插件包

```toml
[project]
name = "mflowy-extra"
dependencies = ["mflowy"]

[project.entry-points."mflowy.plugins"]
"load.super_csv" = "mflowy_extra.loaders:super_csv"   # name = step.module
```

```python
from mflowy.driver.handler import handler
from mflowy.builtin_plugins.middlewares import inject_df


@handler(inject_df)
def super_csv(path, **params): ...
```

- 安装即注册：`uvx --from "mflowy[modeling]" --with mflowy-extra mcpSrv`；同名 `step.module` 覆盖内置实现（info 日志）
- 全新 step 族：参照 `getters.py` 为族写 `Get*/inject*` 对；内核侧只需 entry point 声明，无注册表无配置
- SDK 面 = driver 的 `@handler`/尾链 + 本包注入器；破坏性变更受 CHANGELOG 语义化版本约束
