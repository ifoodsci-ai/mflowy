# Roadmap

功能路线图。已实现能力以 MCP `tools/list` 与模块注册表为准——本表只做能力概览，
不逐项复述参数细节（防漂移）；规划项落地后在 [CHANGELOG.md](../CHANGELOG.md)
记录并从下表移除。

## 已实现

| 能力域   | 覆盖                                                                                          | 代码位置                              |
| -------- | --------------------------------------------------------------------------------------------- | ------------------------------------- |
| 数据载入 | CSV / Excel / Parquet / HTTP / 本地文件 / Python 脚本                                          | `packages/builtin_plugins/mflowy/builtin_plugins/loaders/`                |
| 数据清洗 | 缺失值（删除/填充）、异常值（Z-score / IQR）、特征过滤（方差/相关性/通用）、单位清洗、自定义 Python | `packages/builtin_plugins/mflowy/builtin_plugins/cleaners/`               |
| 特征工程 | 类别编码 ×5、数值变换 ×8（缩放/对数/幂/分箱/PCA/交互特征等）                                   | `packages/builtin_plugins/mflowy/builtin_plugins/x_transformer/`          |
| CV 策略  | 单节点 `split`（`cv` 折数 + `repeats` + `group_by` 三参数自动决策划分族，全对位 sklearn splitter）+ `SplitSpec`（outer holdout 与 inner CV 同源） | `packages/builtin_plugins/model/mflowy/model/cv/`                              |
| 模型算法 | XGBoost、LightGBM、CatBoost、RandomForest、PyTorch MLP（回归/分类/多分类）                     | `packages/builtin_plugins/mflowy/builtin_plugins/model/`                  |
| 超参优化 | Optuna TPE 自动调参                                                                            | `packages/builtin_plugins/mflowy/builtin_plugins/model/study.py`                  |
| 评估指标 | 回归 4（MAE/RMSE/R²/MAPE）+ 分类 13（Accuracy/BalancedAccuracy/Precision/Recall/F1/MCC/Kappa/AUC-ROC/PR-AUC/LogLoss/MLogLoss/Brier/ECE）+ per-class 报告 | `packages/builtin_plugins/mflowy/builtin_plugins/model/types.py`          |
| 可视化   | 数据分析 / 模型评估（含 ROC / PR / 校准曲线 / Bland-Altman / 归一化混淆矩阵）/ SHAP 可解释性三族图表 | `packages/builtin_plugins/mflowy/builtin_plugins/plots/`                  |
| 反向搜索 | 输入寻优（目标最大化/最小化、范围约束、跨列规则）                                               | `packages/builtin_plugins/mflowy/builtin_plugins/model/search_input.py`   |
| 统计     | 数据画像、效应量                                                                                | `packages/builtin_plugins/mflowy/builtin_plugins/statistic/`              |
| 自主研究流（分层落位：π 在 agent，环境面在 MCP） | 工具层：`detect` 检测算子族（missing/duplicates/outliers/constant/stringly/id_like——pattern 已正交三拆，统一错误 schema `ErrorReport`）+ `clean` 修复算子族（动作词表，含 `drop_duplicates`/`range_clip`）+ `statistic.profile` 的 `[findings]` 规则提示 + `NodeResult.value` 结构化通道；应用层：清洗 MDP 会话引擎（start/propose/apply-dry-run/rollback/finish：版本快照 + diff 观测 + 内生质量分 quality_score + 决策日志 (s,a,r,s′) + 导出可执行清洗计划）、`compile_survey`（问卷→DomainCard fail-loud）、`data_profile` findings 编译直达 agent；survey skill 剧本（.agents/skills/survey/SKILL.md，跨宿主中立位）。`clean.auto`/`survey` 节点已退役（π 上移） | `packages/builtin_plugins/data/mflowy/data/detect/`、`packages/builtin_plugins/data/mflowy/data/cleaning_session.py`、`packages/mcp/mflowy/mcp/tools.py` |
| 实验追踪 | MLflow 全量记录（参数/指标/模型/图表）+ 数据血缘 tag（`mflowy.lineage`）                    | `packages/builtin_plugins/data/mflowy/data/middlewares/`       |
| **两 MDP 会话化（共识 v3 · 全量落地）+ modeling 真决策化（#31）** | 六 dist（utils/driver/**data**/**model**/mcp，extras=域选择器 `[data]`/`[model]`/`[mcp]`/`[deep]`）+ mcp 零域依赖（`mflowy.tools` 注册表，域包自持工具）；清洗会话（终局 U + θ 门 + preview + clean_abandon + clean skill，#29/#30 契约合规）；建模会话回路状态机重构（#62–#64——八动作面：配置编辑零步费 + run_hpo 增量 sqlite study/final_train_test 先搜后店+combo 冻结、Δu_inner 奖励、OOF 观察面 + diagnose 四算子 + findings 双层（规则 + report_findings 落诊）、propose 规则引擎重写（fire/prioritize 记分牌）、CVCard 结构化候选、finish 两段式（preview 产物清单 + conclusion → LaTeX PDF）、hyperparameters_search/modeling 工具拆分）+ 控制面（rollback 光标回退/abandon）、族目录开放（discover 过滤 `.mdp_family` 声明者——structure+machinery+regularizers）、五节点 `**_` 删除、model-brief/research-review/model skill；model.auto/clean.auto 内嵌 π 均已退役；知识归位（repair/structure/mdp_family 自声明随词汇主人） | `packages/builtin_plugins/data/`、`packages/builtin_plugins/model/`、`.agents/skills/` |

## 规划中

| 方向              | 说明                                                |
| ----------------- | --------------------------------------------------- |
| **评估不确定性量化** | 选参指标的折间 std 已随训练落 run（`{scoring}_std`，来自 inner CV 分数）；待补：① 全指标折间离散度（终评后补一轮 multi-metric CV，或以 holdout bootstrap 置信区间替代）；② `predict` 输出预测不确定性（单一最终模型下需换来源：conformal prediction 或训练时落盘的 OOF 残差分布）；③ 差距小于折间波动时标记"无显著差异"。证据链：科学审计 P0（排名决策无显著性证据）+ P1（预测无不确定性） |
| **逆向搜索可信度约束** | `search_input` 三项防护：① 置信度约束——优化目标在最终模型输出上附包络外推惩罚项或硬约束（单一最终模型下分歧来源改为训练时落盘的 holdout 残差分布）；② 密度感知搜索空间——推断空间附观测密度加权或显式警告包络外推区域；③ 收敛证据——输出 best-value 轨迹与多种子稳定性对比，替代固定 n_trials 的"预算内最好"。证据链：科学审计 P0（模型即真理/包络幻觉/无收敛证据三大逆向设计经典陷阱均未设防） |
| **效应量检验严谨化** | `effect_size`：① 多重比较 BH 校正（当前按 min p 挑 top-3 无校正）；② η²→ω²、Cramér's V→bias-corrected 版本；③ 方差不齐时 Welch ANOVA。证据链：科学审计 P1；效应量为主 p 为辅的定位不变，校正只影响 p 的使用可信度 |
| **任务推理硬化** | `infer_task_type_by_statistic` 工具与 x_y 内部 `_infer_task` 两处：① unique_ratio 阈值附样本量依赖说明（小样本 r≥0.5 判回归、大样本 r<5% 判分类，同一变量结论随 n 漂移）；② r<5% 短路提前于等距整数甄别——真实整数量（计数/年龄）大样本下误判分类，短路前增加等距整数检查。证据链：科学审计 P1 |
| **族内输出注解一致性（目录面）** | 运行期复核（`_check_value` 撒谎 fail-loud 三处对称）与构建期声明复核（`verify_output_hint`）已随参数边落地；残余在目录面——归组成员返回注解互相一致性仍无校验，第三方 `model.my_model` 声明与族内漂移要到消费点才暴露。方向：测试期加载族成员比对返回注解（原构建期 AST 扫描钩子已随注册静态化退役），坏契约 fail-loud 在目录面 |
| **沙箱节点族泛化** | agent 现场生成计算的通道仅 `load.python`/`clean.python` 两个沙箱节点（外部 .py + 固定 df 契约）；方向：泛化为「声明所属 namespace + 满足该组输出契约」的通用沙箱节点，经 validate 面校验后进图——用族输出契约校验（见上条）换「agent 扩词表」的回路，不引入运行期注册/热重载 |
| 数据合成          | CTGAN + 约束 pipeline                                |
| 超参采样器扩充    | Random / Grid / CMA-ES（当前仅 TPE）                  |
| 缺失值专项可视化  | 当前由数值分布图间接覆盖                              |
| 数据血缘可视化    | 当前为 MLflow tag，规划 UI 视图                       |
| 时间序列支持      | 按需                                                  |
| 英文文档 / 国际化 | 按需                                                  |
| PyPI 发布         | 0.1.2 已实存于 PyPI（2026-08 起）；0.2.0 待发布 |
