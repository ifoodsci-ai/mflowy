# 模型训练任务的 MDP 设计：管线结构与预算分配的序贯决策形式化

> 设计日期：2026-10-09；2026-10-11 修订：终态形态改为**建模会话**（modeling_session，π 上移 LLM）——§2/§4/§5/§8 重写；已落地的 `model.auto` beam 节点降级为参考策略，会话化后退役。姊妹篇（清洗 MDP，同步修订）见 [2026-10-09-cleaning-mdp-design.md](2026-10-09-cleaning-mdp-design.md)。

## 0. 定位：训练任务里序贯结构在哪

MFlowy 建模现状 = 人/agent 选定 x_transformer 集合 + 模型族写进 YAML，超参以 `int | ContinuousSpace[int]` 双态参数声明进签名（定值=冻结、空间=搜），TPE 在**固定结构内**搜参（n_trials=100，pruner 形参存在但全仓无人传——跨 trial 预算分配是空白）。

序贯结构有三层，前两层进本设计：

1. **构型决策序贯且有条件依赖**——变换器的价值以模型族为条件（代码自证：`standard_scaler` docstring「PCA/SVM/KNN/神经网络/逻辑回归的默认前置步骤」；`pca_reducer`「树模型一般不需要」；`interaction_creator`「下游线性模型无法自动捕捉时使用」）。同一数据状态下换了下游模型，动作价值整个翻转。TPE 把结构当固定背景，条件依赖全丢；
2. **预算分配序贯**——n_trials 均匀撒，没有「这族结构没戏早停、有戏加注」的跨评估决策；
3. （显式出范围）**训练内控制**——PBT 式中途调 lr/早停。XGB/LGBM/CAT/RF 非迭代交互式训练，MLP 小规模，v1 不做。

**与清洗 MDP 的分工**：清洗 MDP 改写数据（D_t 演化），建模 MDP 改写构型（数据不动——x_transformer 产出 `(name, transformer, columns)` 三元组而非改写后的 df，fit 发生在折内）。两者是 sibling，经 `DomainCard`（问卷先验）与清洗后 df 组合，**不合并**——数据面动作与构型面动作的状态空间、转移性质（确定 vs 随机）、代价结构完全不同。

## 1. 与清洗 MDP 的三个本质差异

1. **状态不在数据，在构型与搜索历史**。s_t = (φ(D_train) 不变项, 构型 P_t, 评估历史 H_t, 预算 b_t)——H_t 是一等公民：同一构型在「邻居试过很多」与「没试过」下最优动作不同（避重复、聚焦有希望区域）。本质是把**黑盒优化过程本身**形式化为 MDP——TPE 就是这个 MDP 上跑的一个固定贪心策略，形式化把它变成可学/可搜索的。H_t 的物理载体现成：**MLflow runs**——「实验记录即 agent 记忆」在此从隐喻变字面；
2. **转移不再确定**。编辑动作确定（只改构型），评估动作随机：score ~ N(μ(P), σ²(P)/k)。严格说这是 **POMDP**（μ(·) 是隐状态，H_t 是 belief）——工程上按 MDP 处理 belief；
3. **动作代价极不对称**。改构型便宜（改 YAML），评估贵（一次训练 + CV）。**EVAL 成为显式动作**。

## 2. 五元组 M = (S, A, P, R, γ)，视野 = 分档预算（2026-10-11 修订：会话版）

- **S = (D, card, protocol, draft, board, b)**：D = TrainSplit（清洗终态经 x_y/split 物化，
  episode 内冻结）；card = DomainCard 先验；**protocol = (task, metric, CV 切分器)**——
  初始化段（model-brief 问卷 → `compile_model_brief` → `ProtocolCard`）选定后冻结，
  **不进动作空间**：换协议 = board 全部作废（指标不可比）+ test 重触（no-peeking 破防），
  只能 abort 开新 episode（findings 作为先验回流）；draft = 结构草稿（类别编码槽 ×
  数值缩放槽 × 特征构造集 × 对比族集，即 modeling_steps_yaml 的结构化形态）；
  board = 记分板 {(结构×族) → (U_mini, 成本)}——历史动作的信息全部沉淀进 board，
  马尔可夫性由此成立；b = 分档预算 (b_mini, b_full)。**phase 是派生量而非存储**：
  P1 ⟺ board 空 ∧ ¬full_done，P2 ⟺ 有评估 ∧ ¬full_done，P3 ⟺ full_done，T ⟺ finished。
- **A 四档**（动作成本极端异质，分档是建模域与清洗域的最大不同构）：① 协议动作
  （初始化段一次性，规则给候选）；② 结构编辑（零成本，改 draft）；③ **EVAL-mini**
  （中成本：快速配置 inner-CV，U 入 board，不触 test 不进 mlflow）；④ **EVAL-full**
  （终局性：获胜结构委派真实 model 族节点——TPE + 全量重训 + holdout 终评**一次** +
  持久化 name='final'）。元动作：explanation / rollback / finish / abort。
- **P 确定性三型**：编辑 = draft 纯变换；eval_mini = board 增量（固定 seed 可复现）；
  eval_full = ModelLoader 工件 + U_full。守卫前置：相容性、no-peeking、预算、
  ID/protected 禁入特征。**eval_full 是单向门**：穿门后动作空间坍缩
  {explanation, findings, finish}，rollback 不可穿越（门后回退 = 对 test 求梯度）；
  full **失败**不锁死（b_full 已扣、full_done 复位、该结构标记失败，回 P2 换次优重试）。
- **R**：mini 档 r = ΔU*（当前最佳 U_mini 的增量，未超越记 0——探索失败由预算硬约束
  承担，不用负奖励）；复杂度惩罚内嵌于 U（每非空槽 −|U|×0.2%，实现期修正 §6-1）；
  终局 R_T = U_full − λ·复杂度 + **过拟合门**（|U_inner − U_holdout| > δ → findings
  记过拟合信号，可拒收 finish）。与清洗 MDP 的本质差异：R **不是纯内生代理**——U 本身
  就是目标信号，故无需清洗域那种「终局下游奖励」补丁。
- **γ**：不建模（回合制 + 硬预算有限视野，γ=1）；「少评估、早收敛」偏好用省预算奖励
  表达（finish 时剩余 b 折算小额奖励），不用折扣。

## 3. 文法事实：列分区并行，非有序链

`x_preprocessors` 边把全部变换器收进 `ColumnTransformer(remainder="passthrough")`——**列分区并行组合**。「先缩放再 PCA」类序贯模式当前文法不表达（docstring 有建议、结构无载体）。动作空间按实际文法设计：结构 = (类别编码槽 × 数值缩放槽) + 模型族，编辑动作作用于槽位取值；序贯组合是 x_transformer 文法的独立演进项，不属本 MDP。

## 4. 会话化：外层 π = LLM，内层 TPE 不重造（2026-10-11 修订）

落地形态与清洗 MDP 对称：环境面 = modeling_session 引擎（落 mflowy-model 域包），
π = 调用方 LLM 经 MCP 工具驱动回合。原「外层 MDP + 内层 TPE」两层结构保留，但外层
策略从内嵌 beam 改为 LLM——`model.auto` 的 beam 降级为**参考策略**，会话化后退役
（决策权放行原则：每步 (s,a,r,s′) 可审计、可回滚，搜索器只是 π 的一种实现）。
EVAL 两档机器全量复用（fast 配置表 + build_pipeline + cross_val_score + 终局委派真实节点）。

**状态机**（phase 派生自 (board, full_done, finished)，动作掩码 = propose 规则 + 守卫）：

```
P0 参数化段（非 MDP）  model-brief 问卷（规则预填 + 用户确认必问四项：建模成功判据/
  指标偏好/约束（可解释性/预算/禁用族）/外部基线）→ compile_model_brief
  （四项相容校验 fail-loud：task↔target dtype、metric↔task、cv↔数据事实、预算下限）
  → ProtocolCard 注入 modeling_session_start，(D, card, protocol) 冻结
P1 基线段   首动作恒为零 FE × 推荐族的 eval_mini（baseline-first 编码在 propose 而非守卫）
P2 结构搜索段   edit / eval_mini / eval_full / rollback(draft·board 版本，预算不回补) / abort
P3 终局段（eval_full 单向门后，动作空间坍缩）   explanation / findings / finish（过拟合门可拒收）
T 终态   导出 modeling_steps_yaml（可独立重放）+ ModelLoader + 记分板 + findings
```

**规则引擎**（三类规则，知识随词汇主人居住，不做集中引擎；输出=候选+缺省，最终选择权留 π）：

1. **候选生成（propose）**：CV 协议规则（规模×形状→档位候选，住 cross_validation：
   <1k→repeated/LOOCV、1k–100k→5-fold 缺省、>100k→holdout 优先；不平衡→Stratified、
   组列→GroupKFold、时序→TimeSeriesSplit）；相容性矩阵（族×编码×缩放合法组合，
   各模型族自声明 `cat_support`/`scaling_need`/`min_samples`——`model.auto._valid` 的
   泛化与下沉）；baseline 推荐（task+规模→族提名，住各族）；特征构造规则
   （dtype/分布/基数→候选，住 x_transformer 各模块）；
2. **守卫（转移门）**：no-peeking（test 只在 eval_full 触碰一次，其后动作空间坍缩；
   board 只存 U_mini，U_full 只进终局）；防泄露（target-aware 变换折内拟合断言；
   ID/protected 禁入特征矩阵，来自 DomainCard）；合法性（相容性未命中拒、样本量下限）；
   预算（b_mini/b_full 分档硬顶）；
3. **评分与 findings（R 侧与观测）**：指标映射（infer_metric + higher_is_better +
   mean−std 聚合）；复杂度惩罚表；**建模域 findings 词表**（首批 3 kind：过拟合信号/
   切片塌陷/误差集中列——ErrorReport 的建模域对应物，独立评估动作的产物，也是回流
   清洗域的载体；校准偏差与 baseline 优势 kind 后补）。

**判别式**：凡可用数据事实确定性推出的 → 规则引擎；凡需评估结果才能分辨的（编码
最终选择、超参取值）→ 搜索/π。规则缩小动作空间，不消灭决策空间。

## 5. MFlowy 映射（2026-10-11 修订：会话版）

| MDP 元素 | 已有件（白嫖） | 实现增量 |
| --- | --- | --- |
| 状态 D | 清洗会话终态 parquet + `x_y`/`split` + SplitSpec | — |
| protocol | `infer_task_type_by_statistic` + split 三参数 | model-brief skill + `compile_model_brief` + `ProtocolCard`（落 mflowy-model） |
| 动作目录 | x_transformer 13 节点 + model 5 族（fast 配置表） | modeling_session 引擎（状态机/版本树/决策日志）+ mcp 工具壳 |
| EVAL-mini | `model.auto` 的快速配置 + `build_pipeline` + `cross_val_score` 机器 | 会话化复用（board 记分） |
| EVAL-full | `training()`（TPE + holdout 唯一触碰 + ModelLoader + 遥测） | 终局委派同款（单向门 + 过拟合门） |
| 动作合法性 | `model.auto._valid`（sklearn 系族 × passthrough 类别列） | 相容性下放各族声明字段（cat_support/scaling_need/min_samples） |
| H_t / 记忆 | MLflow runs + 血缘 + `list_runs`/`get_run` | board（episode 内）+ 记分板导出（跨 episode） |
| findings | ErrorReport / quality_score（数据域同构件） | 建模域 findings 词表（3 kind）+ 独立评估动作（消费 ModelLoader，不付重训代价） |
| baseline | ①现状（人/agent 定结构 + TPE）②TPE 联合搜 ③AutoGluon/FLAML | P1 基线段强制首步（零 FE × 推荐族） |

## 6. 实现期修正（搜索版落地时对设计的修正）

1. **λ 惩罚量纲问题**：终局 U 的 λ_len 绝对惩罚与 CV 指标量纲不可通约 → 改为**相对惩罚**（|U| 的 0.2%/槽位）；清洗 MDP 同理改为**硬约束**（行损失 ≤30%、列损失 ≤50%）+ no-harm gate；
2. **EVAL-mini 用固定快速配置**而非小 TPE study——省 optuna 开销且无 mlflow 依赖，代价是默认超参下的结构排序可能与调参后不同（successive halving 固有取舍，由 EVAL-full 终裁对冲）；
3. **终局委派**：获胜结构构建 ColumnTransformer 后直接调该族真实节点 pyfunc（同一 mlflow parent run 内嵌套 trial runs），lineage/图表/ModelLoader 全部复用——「探索在内存、终局入图」的节点化等价物；
4. **test 只触碰一次**：mini 评估全部走 inner CV（train 侧），holdout 只在委派终训里由 `training()` 触碰一次。

## 7. 求解路径与陷阱（要点，2026-10-11 修订）

1. 落地序：LLM-π 会话版（modeling_session，本设计终态）→ 参考策略对照（`model.auto`
   beam = 内嵌 π 的历史形态，会话化后退役）→ TPE 联合搜（诚实对照）→ RL/元学习
   （π(编辑 | φ(D), P, H 摘要) 跨数据集摊销，文献格位 AlphaD3M/PD-GMN）——RL 的唯一
   立项依据是跨数据集摊销；**mflowy 特有优势：MLflow 历史 runs 是现成离线语料**；
2. 陷阱：胜者诅咒（mini 粗筛排序偏差由 EVAL-full 终裁对冲 + holdout 对照）、结构过拟合
   （复杂度相对惩罚 + 小数据限视野）、test 贞操（no-peeking 单向门 + board/U_full 隔离）、
   列重叠（ColumnTransformer 允许同列多变换器，复杂度惩罚记账）、词表漂移（动作目录
   动态构建，新算子冷启动）。

## 8. 验收（2026-10-11 修订：会话版）

- **自动化 E2E**：装配链 load → survey 问卷 → 清洗会话（start/propose/apply/finish）→
  model-brief 问卷（compile_model_brief）→ modeling 会话（start/propose/apply/
  eval_full）→ SHAP，全程断言守卫与 no-peeking 不变量；
- **challenge 人工回放**：以 `examples/` 挑战赛数据在新会话引擎上重走，暴露 propose
  质量、观测 token 经济等只有真实回合才可见的问题；
- 参考策略对照（对 `model.auto` / TPE 联合搜的胜负证据、holdout 与 inner-CV 差值并列
  报告）保留为 **RL 学习化的立项前提**，不阻塞会话版落地。
