# 数据清洗 MDP 设计：清洗链的序贯决策形式化

> 设计日期：2026-10-09；2026-10-11 修订（§7）：落地形态为**会话版**（π 上移 LLM），`clean.auto` 搜索版节点废弃，终局 U / θ 验收门转二期。本文是带日期的快照式设计文档：五元组数学为终态，与代码的映射表以设计当日实况为准（cleaner 签名、profile schema 等随代码演进，以 [packages/builtin_plugins/README.md](../../packages/builtin_plugins/README.md) 与 MCP `list_nodes` 为准，不复述）。立项条目见 [docs/roadmap.md](../roadmap.md)「两 MDP 优化」。

## 0. 定位：为什么形式化成 MDP

清洗本质是**序贯决策**：动作效果依赖前序动作——先 winsorize 再中位数填充 ≠ 先填充再 winsorize（填充统计量被异常值污染）；异常值处理改变相关性判据，进而改变 `correlation_filter` 的删除集。现有两条路径都没有刻画这种序贯结构：

- **LLM agent 现场编排**（现状）：同一个 MDP 的 zero-shot 策略——状态靠 profile、动作靠 clean 目录，但不可复现、不可审计、无 credit assignment；
- **Optuna TPE 联合搜链**：把整条链当一个 trial 黑盒优化，trial 之间不共享「哪一步好」的结构信息，搜索空间随链长组合爆炸。

MDP 形式化把每步决策变成可赋 credit 的单元。它同时是仓库两条独特叙事（[2026-10-09 价值评估](2026-10-09-value-assessment.md) §4）的合流点：**动作空间 = 闭环词表**（`clean` 目录），**轨迹记录 = 实验记忆**（每动作一个 MLflow run + `df_diff` + 血缘）。

## 1. 五元组 M = (S, A, P, R, γ)，有限视野 T

### 状态 S：s_t = (φ(D_t), t)

φ 是画像特征提取，列级字段直接映射 `statistic.profile` 已有 schema：

- **列级**（变长集合）：`missing` 比例、`nunique`、`cardinality_ratio`、`is_id` / `is_constant`、dtype one-hot、`skew`（绝对值）、`kurt`、四分位归一化 spread、异常值比例（|z|>3 占比）
- **全局**：shape、重复行率、类型计数、高相关列对数（ρ>0.9）、近零方差列数、目标列缺失率
- **预算**：t/T
- **编码**：变长列集合用 DeepSets/SetTransformer 聚合，或取 top-k 最差列（按缺失率、|skew|、异常比例排序）拼接，附列身份 embedding

**Markov 性**：转移确定、奖励只依赖 (D_t, a_t) 与终局 → **(D_t, t) 是充分统计量**，数据状态本身即 Markov——已执行动作历史不是必需。但 φ 是有损压缩，不同 D 可产生相同画像（aliasing），工程补救：列身份 embedding + 最近 k 步动作计数作网络输入（frame stacking，加速学习，不承担 Markov 义务）。

### 动作 A：参数化动作空间

**A = ⋃_{op ∈ 目录 ∪ {STOP}} {op} × Θ_op**

- **算子层** = `clean` 目录动态枚举（词表即动作空间，第三方清洗插件安装即扩动作空间；参数 schema 从 Annotated/docstring 编译，与 MCP schema 同源）：

| op | 参数层 Θ_op |
| --- | --- |
| `drop_missing` | threshold ∈ [0.3, 0.9] 网格 |
| `fill_missing` | strategy ∈ {mean, median, interpolate, forward, backward, mode, fill, fill_grouped_mode} × 目标列 × missing_indicator |
| `zscore_detector` / `iqr_detector` | threshold × strategy ∈ {remove, cap, replace} × 检测列 |
| `variance_filter` / `correlation_filter` | threshold |
| `strip_units` | 列 × 后缀（枚举画像中检出带单位后缀的列） |
| `common_filter` | remain/drop 列集（受限于 top-k 候选） |
| `STOP` | — |

- **列选择子空间**（columns 参数维度 = 当前列数，变维）：三档处理——①候选集压缩：动作只允许作用于状态中 top-k 异常列（把 2ⁿ 压到 k）；②pointer/attention 网络选列；③粗粒度档位（所有数值列 / 所有高缺失列）。**起步用 ①**，与画像排序天然对齐。
- **混合离散-连续空间三档**：(a) 参数全离散化进动作（起步，DQN 可解）；(b) PAMDP 算法（P-DQN / MP-DQN，Q(op, x) 把连续参数作条件输入）；(c) SAC + hybrid head。
- **`clean.python` 排除出动作集**：自由代码不是良构动作，且与闭环词表立场冲突——留作专家先验 / 沙箱通道，不进策略。
- **重复/无效动作不禁止**：环境照常执行（多为 no-op），由步成本惩罚消化。比非法动作掩码简单，且不破坏 MDP 结构。

### 转移 P：确定性（本问题最大的结构红利）

**D_{t+1} = f_{a_t}(D_t)**，f 即 clean handler——df→df 纯函数，填充值、阈值、过滤集全部是当前数据的确定函数。P 退化为 δ 点质量：

- Q 学习目标无转移噪声（TD 目标里没有 P 的方差）；
- 可精确 undo（df 快照或按血缘重放）；
- **model-based 在线规划可行**（beam search / MCTS / rollout）——确定性 + 小 T 下根本不用「学」，见 §4；
- 随机性只有两处，都不在 P 里：s₀（哪个数据集）和 R 的评估噪声。

转移观测面已存在：所有 cleaner 挂的 `df_diff` 中间件，每步 df 差异本来就在落盘。

### 奖励 R：势函数整形 + 终局真值

**终局**（STOP 或 t=T 时）：

```
U(D_T) = M(D_T) − β·σ(D_T) − λ_r·(1 − m_T/m_0) − λ_c·(1 − n_T/n_0)
```

- M = inner CV 平均分、σ = 折间 std——直接复用 `split`（SplitSpec）+ `model` 族已落 run 的 `{scoring}_std`（roadmap「评估不确定性量化」在此接上）；β 取 1–2 倍折间 std 权重，防策略追评估噪声；
- 数据损失惩罚 λ_r、λ_c **必须非零**：这是对 reward hacking 的主防线——CV 分数对「删掉难洗的列」可能不降反升（删噪声列是真增益，删信息列短期未必掉分），没有列惩罚，最优策略会退化成激进删列。

**每步**：

```
r_t = −c + γ·Φ(s_{t+1}) − Φ(s_t),  Φ(s_T)=0
```

Φ(s) = 数据质量势函数（负缺失率、负异常比例、负 |skew| 的加权聚合，权重手工设定或从历史 run 学）。**关键理论约束：必须用 potential-based shaping**（Ng et al. 1999）——`γΦ(s′)−Φ(s)` 形式保证最优策略不变；若直接把 Δquality 加进奖励，会引导策略停在「画像好看但模型不涨」的状态，扭曲目标。c 为常数步成本，激励短链。

### γ 与初始分布

- 有限视野 T = 10–20（或 2×严重列数）；γ=1 在有限视野下良定义，但稀疏终局奖励下实践取 γ≈0.95 更稳；
- s₀ ~ 数据集总体，**一个 episode = 一个数据集**；训练时行 bootstrap + 列子集做增广，防策略过拟合单一数据集。

## 2. 防泄露的环境结构（不是可选项）

对齐仓库「x_transformer 训练数据内 fit 防泄露」的既有哲学：

1. episode 开始先出 `SplitSpec`；**D_t 全程是 train fold 的演化**，状态 φ 只看 train fold，test 不进任何奖励回路，只用于最终报告；
2. 清洗统计量（填充值、winsorize 边界）在演化中的 D_t 上计算——这是「序贯清洗」的固有语义；
3. 诚实声明一个偏差：清洗决策在 CV fold 外做出、全折共享，清洗参数实际处于「超参」地位——评估仍诚实（没碰 test），但严格版（nested：每折内重放 MDP）成本平方倍，默认不做，设计点名即可。

## 3. MFlowy 映射：哪些白嫖、哪些是增量

| MDP 元素 | 已有件（白嫖） | 新增增量 |
| --- | --- | --- |
| 动作空间枚举 | `clean` 目录（`list-nodes` / `get-node-info`），参数 schema 与 MCP schema 同源 | 参数离散化档位表 |
| 转移算子 f_a | clean handler + `df_diff` 遥测 + 每节点 run | — |
| 状态 φ | `statistic.profile` schema | 数值化编码 + 全局量（重复率、高相关对数）+ 列身份 embedding |
| 终局评估 U | `split` + `model` 族 + inner CV + `{scoring}_std` | 奖励组装（λ、β 权重） |
| 轨迹记录 | MLflow 血缘（`mflowy.lineage`）——清洗链即天然 trajectory log，可按血缘重放 | — |
| 执行位置 | JobProvider 委派（策略训练是长任务）；推理期可本地 | 训练循环编排 |
| baseline | Optuna TPE（链模板联合搜）、人工规则链（research-flow 顺序：缺失→异常→过滤）、空清洗 | — |

真正的增量只有三件：**gym 风格环境包装器**（`reset(dataset) → s₀`，`step(a) → (s′, r, done)`，内部走 DAG 执行 clean handler）、**φ 编码器**、**策略侧**（搜索器或网络）。数学先立，实现落点（新插件族 vs JobProvider 新方法）是后续决策。

## 4. 求解路径：搜索优先，学习看泛化

按成本递增，且**前者是后者的 baseline**：

1. **贪心/规则链**（先有）：固定方法论顺序，每步枚举候选动作取 ΔΦ 最大；
2. **在线规划**（性价比最高）：确定性转移 + 小 T → beam search / 浅 MCTS 直接规划，单数据集场景通常优于学习，**无需训练**；
3. **TPE 联合搜链**：Optuna 已有，无 credit assignment 但样本效率未必输 RL；
4. **无模型 RL**：参数离散化 → DQN；连续参数 → P-DQN / SAC hybrid。

**关键判断**：RL 的价值主张不在单数据集（那里搜索就够），而在**跨数据集泛化**——画像特征是数据集无关坐标系，学出的 π(a | φ(D), t) 在新数据集上零搜索成本（amortized search）；Optuna 和在线规划都不迁移。roadmap 立项依据是「多数据集场景的摊销收益」，单数据集证据用搜索版就能拿到。

## 5. 陷阱清单

1. **删列提分**（reward hacking）：λ_c 惩罚 + 产出报告并列「数据保留率」；
2. **评估噪声**：单次 CV 分数方差大 → mean − β·std，β≥1；
3. **状态 aliasing**：画像压缩丢信息 → 列身份 embedding + 近期动作特征；
4. **动态列空间**：`fill_missing(missing_indicator=True)` 会加列、过滤会删列——set 编码天然支持，固定维拼接需 padding；
5. **词表漂移**：第三方 clean 插件安装后动作空间变化 → 动作空间从目录动态构建，新算子冷启动（不做 zero-shot 泛化声明）；
6. **proxy 与真目标错位**：Φ 的手工权重可能引偏——shaping 只许加速、终局定胜负，定期用真值校准 Φ；
7. **可交换冗余**：互不相干列的动作可换序，同一链有 k! 等价排列——策略输入用「已执行动作集合」不变表示或接受冗余。

## 6. 立项前提与 MVP（2026-10-11 修订：已被会话版取代）

原 MVP 定义（§4-2 搜索版 + gym 环境包装器 + 贪心/beam）**未按原形落地**——实施期决策
改为把每步决策权放行给调用方 LLM（π 上移）：环境做成会话引擎，策略由 agent 经 MCP
工具驱动（见 §7）。搜索版/RL 的立项依据不变（跨数据集摊销），作为参考策略后置。

## 7. 落地形态（2026-10-11）：会话版与五元组映射

**π 上移**：落地形态不是 §4 的任何内嵌求解器，而是「环境在引擎、π 在调用方」——
清洗 MDP 以会话引擎落地（`packages/mcp/mflowy/mcp/cleaning_session.py`，规划迁入
mflowy-data 域包）：环境面 = 转移/观测/奖励/预算，π = LLM 经 `clean_session_start /
clean_propose / clean_apply / clean_rollback / clean_finish` 五工具驱动回合。

| 元素 | 会话版实现 | 与 §1 数学的差异 |
| --- | --- | --- |
| S | (当前版本 parquet, 冻结 DomainCard, 剩余预算)；π 的决策输入 = digest(kind×count) + 质量分 + diff | φ 从可学习数值特征退化为规则摘要（π 是 LLM，无需编码）；历史只进决策日志不进决策输入，马尔可夫性成立 |
| A | 参数化 clean 词表 ∪ {rollback(k), finish}；`clean_propose` = 合法走法生成器（错误类型 → 参数化候选 + rationale，LLM 只做选择）；dry_run 是不占步的假想转移 | §1 的 top-k 列压缩由 propose 内建（每 kind 取前 5 列）；`clean.python` 同样排除 |
| P | 确定性直调（词表动作 raw pyfunc）+ 守卫前置（never_drop 存活、目标列值禁改） | §1-3 的确定性判断成立并被兑现为「决策日志可全量重放」（finish 导出的 plan 入 DAG 得到同一终态） |
| R | r = Δquality（`quality_score` 规则加权 Φ）；守卫违约 = 转移被拒（无 r） | λ_r/λ_c 软惩罚降级为硬守卫 + info_loss 观测；步成本 c 外生为 max_steps 硬预算；**终局 U 未接**（下游信号单向，不进会话奖励）——二期补 finish 终局 U（train-fold CV 一次，堵「过度删行刷质量分」） |
| γ | 未建模：回合制 + max_steps 有限视野 ⇒ γ=1，无早修偏好 | 势函数整形的理论约束（保最优策略不变）在 LLM-π 下退化为提示词层面的「读 Δquality 决策」 |

二期方向（roadmap「两 MDP 优化」）：finish 终局 U、θ 验收门（finish 可拒收返回继续）、
规则知识归位（kind→候选映射从会话引擎下放词汇主人模块）、先验→规则热更新（声明式
规则对象；触发条件：真实改规则的频率出现）。

## 来源清单（设计当日代码实况）

- `packages/builtin_plugins/mflowy/builtin_plugins/cleaners/`——9 个 clean 节点签名（drop_missing / fill_missing / zscore_detector / iqr_detector / variance_filter / correlation_filter / common_filter / strip_units / python）；全部挂 `df_diff` 中间件；fill_missing 策略枚举八种
- `packages/builtin_plugins/mflowy/builtin_plugins/statistic/profile.py`——画像 schema（missing / nunique / cardinality_ratio / is_id / is_constant / skew / kurt / top_10）
- `packages/builtin_plugins/mflowy/builtin_plugins/middlewares/log_df_diff.py`——每步 df 差异遥测（转移观测面）
- `packages/builtin_plugins/mflowy/builtin_plugins/cross_validation/`——`split` / SplitSpec（outer train/test + inner CV 同源）
- `{scoring}_std` 折间 std 已随训练落 run——[docs/roadmap.md](../roadmap.md)「评估不确定性量化」条目
- `packages/builtin_plugins/mflowy/builtin_plugins/model/study.py`——Optuna TPE（baseline 基础设施）
- [packages/driver/README.md](../../packages/driver/README.md)——x_transformer「训练数据内 fit 防泄露」哲学、血缘 `mflowy.lineage`
- [2026-10-09-value-assessment.md](2026-10-09-value-assessment.md)——「闭环词表 + 实验记录即记忆」双叙事（§0 合流点）
