# Roadmap

功能路线图。已实现能力以 MCP `tools/list` 与模块注册表为准——本表只做能力概览，
不逐项复述参数细节（防漂移）；规划项落地后在 [CHANGELOG.md](../CHANGELOG.md)
记录并从下表移除。

## 已实现

| 能力域   | 覆盖                                                                                          | 代码位置                              |
| -------- | --------------------------------------------------------------------------------------------- | ------------------------------------- |
| 数据载入 | CSV / Excel / Parquet / HTTP / 本地文件 / Python 脚本                                          | `packages/builtin_plugins/src/mflowy/builtin_plugins/loaders/`                |
| 数据清洗 | 缺失值（删除/填充）、异常值（Z-score / IQR）、特征过滤（方差/相关性/通用）、单位清洗、自定义 Python | `packages/builtin_plugins/src/mflowy/builtin_plugins/cleaners/`               |
| 特征工程 | 类别编码 ×5、数值变换 ×8（缩放/对数/幂/分箱/PCA/交互特征等）                                   | `packages/builtin_plugins/src/mflowy/builtin_plugins/x_transformer/`          |
| CV 策略  | Simple / K / Stratified / Group / StratifiedGroup K-Fold、Repeated(K/Stratified)、LeaveOneOut、LeaveOneGroupOut | `packages/builtin_plugins/src/mflowy/builtin_plugins/cross_validation/`       |
| 模型算法 | XGBoost、LightGBM、CatBoost、RandomForest、PyTorch MLP（回归/分类/多分类）                     | `packages/builtin_plugins/src/mflowy/builtin_plugins/model/`                  |
| 超参优化 | Optuna TPE 自动调参                                                                            | `packages/utils/src/mflowy/utils/study.py`                  |
| 评估指标 | 回归 4（MAE/RMSE/R²/MAPE）+ 分类 7（Accuracy/Precision/Recall/F1/AUC/LogLoss/MLogLoss）        | `packages/builtin_plugins/src/mflowy/builtin_plugins/model/types.py`          |
| 可视化   | 数据分析 / 模型评估 / SHAP 可解释性三族图表                                          | `packages/builtin_plugins/src/mflowy/builtin_plugins/plots/`                  |
| 反向搜索 | 输入寻优（目标最大化/最小化、范围约束、跨列规则）                                               | `packages/builtin_plugins/src/mflowy/builtin_plugins/model/search_input.py`   |
| 统计     | 数据画像、效应量                                                                                | `packages/builtin_plugins/src/mflowy/builtin_plugins/statistic/`              |
| 实验追踪 | MLflow 全量记录（参数/指标/模型/图表）+ 数据血缘 tag（`mflowy.input_steps`）                    | `packages/builtin_plugins/src/mflowy/builtin_plugins/middlewares/mlflow_log.py`       |

## 规划中

| 方向              | 说明                                                |
| ----------------- | --------------------------------------------------- |
| **评估不确定性量化** | Taylor 排名补充折间离散度：按 (model, fold) 逐折计算指标后输出 mean±std（当前池化全折只出点估计），差距小于折间波动时标记"无显著差异"；`predict` 同步输出 fold 间分歧（回归=预测 std，分类=proba 熵）——点预测当前掩盖置信度。证据链：科学审计 P0（排名决策无显著性证据）+ P1（预测无不确定性） |
| **逆向搜索可信度约束** | `search_input` 三项防护：① 置信度约束——优化目标改用 fold 集成均值，并附 fold 分歧惩罚项或硬约束（高分歧区域不得入选）；② 密度感知搜索空间——推断空间附观测密度加权或显式警告包络外推区域；③ 收敛证据——输出 best-value 轨迹与多种子稳定性对比，替代固定 n_trials 的"预算内最好"。证据链：科学审计 P0（模型即真理/包络幻觉/无收敛证据三大逆向设计经典陷阱均未设防） |
| **效应量检验严谨化** | `effect_size`：① 多重比较 BH 校正（当前按 min p 挑 top-3 无校正）；② η²→ω²、Cramér's V→bias-corrected 版本；③ 方差不齐时 Welch ANOVA。证据链：科学审计 P1；效应量为主 p 为辅的定位不变，校正只影响 p 的使用可信度 |
| **任务推理硬化** | `infer_task` 两处：① unique_ratio 阈值附样本量依赖说明（小样本 r≥0.5 判回归、大样本 r<5% 判分类，同一变量结论随 n 漂移）；② r<5% 短路提前于等距整数甄别——真实整数量（计数/年龄）大样本下误判分类，短路前增加等距整数检查。证据链：科学审计 P1 |
| 数据合成          | CTGAN + 约束 pipeline                                |
| 超参采样器扩充    | Random / Grid / CMA-ES（当前仅 TPE）                |
| 分类评估图扩充    | ROC / PR 曲线、校准曲线（当前仅混淆矩阵）           |
| 缺失值专项可视化  | 当前由数值分布图间接覆盖                            |
| 数据血缘可视化    | 当前为 MLflow tag，规划 UI 视图                     |
| 时间序列支持      | 按需                                                |
| 英文文档 / 国际化 | 按需                                                |
| PyPI 发布         | 当前为本地路径 / 源码方式安装                       |
