# Roadmap

功能路线图。已实现能力以 MCP `tools/list` 与模块注册表为准——本表只做能力概览，
不逐项复述参数细节（防漂移）；规划项落地后在 [CHANGELOG.md](../CHANGELOG.md)
记录并从下表移除。

## 已实现

| 能力域 | 覆盖 | 代码位置 |
| ------ | ---- | -------- |
| **数据清洗子系统（Clean）** | 数据载入（CSV/Excel/Parquet/HTTP/Python 脚本）、画像与 EDA、缺陷检测算子族（DefectType 封闭枚举）、清洗算子族（词表开放、entry points 注册）、清洗 POMDP 会话（六操作、版本树、终局 U + θ 门、决策日志、可重放清洗计划）、DomainCard 先验分级（survey 剧本编译） | [data 域 README](../packages/builtin_plugins/data/README.md) |
| **预测模型训练子系统（Model）** | CV 策略（split / SplitSpec）、特征工程、模型族（XGBoost / LightGBM / CatBoost / RandomForest / MLP，本地建模重栈住 model 包 `[local]`）、评估与解释、建模 POMDP 会话（八动作面、combo 冻结、OOF 观察面、diagnose、findings、finish 收编）、compute 工具（modeling / hyperparameters_search / explanation / predict / inverse_optimization，JobProvider 委派）、model-brief / research-review 剧本 | [model 域 README](../packages/builtin_plugins/model/README.md) |
| **引擎（driver）** | YAML→DAG 增量编译（编译即校验）、平铺键控叙事、参数边三层寻址、LIFO 调度 + 失败血缘剪枝、@node/@edge 插件 SDK、POMDP 会话基座（壳 = 机制，域 Handle = 业务接口） | [driver README](../packages/driver/README.md) |
| **平台（mcp）** | MCP server（stdio）/ JSON runner / 直接 import 三入口、平台工具组、遥测与同意门、mlflow 观测、OAuth 2.1 登录、会话回收 | [mcp README](../packages/mcp/README.md) |
| **底座（utils）** | mlflow 封装、jinja 沙箱、脚本安全门、stdout 捕获、原子写；实验追踪（MLflow 全量记录 + 数据血缘 tag） | [utils README](../packages/utils/README.md) |
| **研究流剧本** | survey / clean / model-brief / model / research-review——全部 `disable-model-invocation`（人点火） | [`.agents/skills/`](../.agents/skills/) |

## 规划中

| 方向              | 说明                                                |
| ----------------- | --------------------------------------------------- |
| **评估不确定性量化** | 选参指标的折间 std 已随训练落 run（`{scoring}_std`，来自 inner CV 分数）；② predict 区间已落地（#164：OOF 残差分位半宽 α=0.1——run_hpo 四面证据 `oof_y_pred.parquet` 即升级原料，CV+/jackknife+ 平滑升级路径见 #190 雾区）；待补：① 全指标折间离散度（终评后补一轮 multi-metric CV，或以 holdout bootstrap 置信区间替代）；③ 差距小于折间波动时标记"波动主导差距"（语义已裁 #80，挂层未决）。证据链：科学审计 P0（排名决策无显著性证据）+ P1（预测无不确定性）。亦是贝叶斯优化推荐子系统（BO，#171）的 Model→BO 接口上游依赖（uncertainty——依赖链评论见 #76；OOF 四面已落 = BO 预测范围对带/配对反馈分母的原料就位） |
| **逆向搜索可信度约束** | 三项防护随 [#82](https://github.com/ifoodsci-ai/mflowy/issues/82) 划归 BO 图 [#171](https://github.com/ifoodsci-ai/mflowy/issues/171)（2026-11-11 起进图内消化）：① 置信度约束——[#174](https://github.com/ifoodsci-ai/mflowy/issues/174) 已裁：BO v1 置信度列 not_available，上游请求 CV+ 逐点区间 + OOD 标记（依赖链评论见 #76）；② 密度感知——原拟 GPSampler 后验天性承接，随 [#175](https://github.com/ifoodsci-ai/mflowy/issues/175) TPE 单引擎裁定失去载体（待真实需求再议）；③ 收敛证据——#171 雾区（毕业条件 = 首版 e2e 验收）。one-shot `inverse_optimization` 维持现状待 BO 图统一重构 |
| **效应量检验严谨化** | `effect_size`：① 多重比较 BH 校正（当前按 min p 挑 top-3 无校正）；② η²→ω²、Cramér's V→bias-corrected 版本；③ 方差不齐时 Welch ANOVA。证据链：科学审计 P1；效应量为主 p 为辅的定位不变，校正只影响 p 的使用可信度 |
| **任务推理硬化** | `infer_task_type_by_statistic` 工具与 x_y 内部 `_infer_task` 两处：① unique_ratio 阈值附样本量依赖说明（小样本 r≥0.5 判回归、大样本 r<5% 判分类，同一变量结论随 n 漂移）；② r<5% 短路提前于等距整数甄别——真实整数量（计数/年龄）大样本下误判分类，短路前增加等距整数检查。证据链：科学审计 P1 |
| **族内输出注解一致性（目录面）** | 构建期声明复核（`verify_output_hint`）已随参数边落地（运行期值级复核随 #57 退役——静态安全哲学）；残余在目录面——归组成员返回注解互相一致性仍无校验，第三方 `model.my_model` 声明与族内漂移要到消费点才暴露。方向：测试期加载族成员比对返回注解（原构建期 AST 扫描钩子已随注册静态化退役），坏契约 fail-loud 在目录面 |
| **沙箱节点族泛化** | agent 现场生成计算的通道仅 `load.python`/`clean.python` 两个沙箱节点（外部 .py + 固定 df 契约）；方向：泛化为「声明所属 namespace + 满足该组输出契约」的通用沙箱节点，经 validate 面校验后进图——用族输出契约校验（见上条）换「agent 扩词表」的回路，不引入运行期注册/热重载 |
| **工业实验闭环（PILOT）** | 贝叶斯优化推荐子系统（BO，[决策图 #171](https://github.com/ifoodsci-ai/mflowy/issues/171) 进行中）+ 学习门禁 + 先验继承链——三子系统闭环 `Clean→Model→BO→Wet-lab↺`，系统设计见 [bo 包 README](../packages/builtin_plugins/bo/README.md)；搜索空间按连续+离散/组合+描述符混合一等公民收口 |
| **BO 多目标扩展** | 产率×纯度 Pareto 等双目标（首版单目标已裁 [#174](https://github.com/ifoodsci-ai/mflowy/issues/174)）；TPE/NSGAII 多目标能力在位（[#175](https://github.com/ifoodsci-ai/mflowy/issues/175)）、扩展不换引擎；触发 = 真实双目标需求且上游供多目标列（[追踪 #188](https://github.com/ifoodsci-ai/mflowy/issues/188)） |
| 数据合成          | CTGAN + 约束 pipeline                                |
| 超参采样器扩充    | Random / Grid / CMA-ES（当前仅 TPE）                  |
| 缺失值专项可视化  | 当前由数值分布图间接覆盖                              |
| 数据血缘可视化    | 当前为 MLflow tag，规划 UI 视图                       |
| 时间序列支持      | 按需                                                  |
| 英文文档 / 国际化 | 按需                                                  |
| PyPI 发布         | 0.1.2 已实存于 PyPI（2026-08 起）；0.2.0 待发布。发版轮镜像策略（[#187](https://github.com/ifoodsci-ai/mflowy/issues/187) 统一版本钉）：同批 wheel 构建 `training-job:<版本>` → k3s 导入（dev 测试位，链中途禁重导）→ 上游种子钉版 → E2E 全链验收 → push `images.mflowy.com/training-job:<版本>`（不可变 tag，永不重推）→ PyPI；mlflow 升 minor 时镜像与客户端同步发 |
