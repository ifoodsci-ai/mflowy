---
name: model
description: 建模训练 MDP 回合操作手册(mflowy)。ProtocolCard 编译后开始建模回合，或建模会话进行中(收到 session_id)时使用。配合 mflowy MCP 工具 modeling_session_start / modeling_propose / modeling_apply(node) / modeling_rollback / modeling_finish / modeling_abandon 驱动回合循环——动作空间 = model 族节点 ∪ FE 节点，环境面在引擎，π(策略)是你。前置:model-brief skill 已产出 ProtocolCard（含 research-review 候选收窄）。
---

# Model Skill：建模 MDP 回合循环（两段式真决策）

三段式研究流第三段。你的职责是**策略（π）**：读 board（u_inner 记分牌）、决定跑哪个
候选、分配搜索深度、决定回滚与交付集。环境面（run 两段式/预算/no-peeking 守卫/
决策日志）在会话引擎。**test 全程只在 finish 被读（每交付模型恰一次）**——run 阶段
你看到的永远只是 inner 证据。

## 开局

```
modeling_session_start(final_path, card=<DomainCard>, protocol_card=<ProtocolCard>)   # 唯一主通道
```

- 协议由 **model-brief skill**（含 research-review 文献候选收窄）产出——candidate_families
  已定第一轮淘汰名单；start 即 D 物化冻结 + 单一预算起算；换协议 = 新 episode（弃局重开）。

## 回合：读证据 → 分配预算 → apply(node, params)

`modeling_apply(node, params)` 与 clean_apply 同形——node ∈ **model 族节点**（run
语义）∪ **FE 节点**（pipeline 槽位语义）。`modeling_propose` 菜单 = 模型候选在前、
FE 候选在后（先定模型、后为它配 FE；evidence 自带**空间摘要**——动态维/静态常量全浮出）：

1. **未跑族**：`modeling_apply("RF", {...})`——搜索（TPE 于签名空间）→ θ* + u_inner
   → 终训 → loader 落盘；响应 `reward` = u_inner（直接读数，跨候选可比）；encode 族 ×
   类别数据时 run 自带最小装配（ordinal）；资源（树数/epoch）= 普通参数，pin 即深。
2. **已跑族的加深题**：折间 std 高 → 重跑加深（n_trials↑，各吃一步）；
3. **FE 动作**（模型选定后出现）：`modeling_apply("onehot_encoder")` 换档 leader
   的非冗余槽位（leader structure 声明冗余的槽不出现），换 FE 后重跑出新板面条目——
   **board 键 = (family, FE 组合)**，不同 FE 下成绩并排可比。

**预算即决策**：单一 max_steps，跑哪些候选、每个搜多深（n_trials 是深度旋钮）由你
分配——好钢用在 board 最有希望的方向。**自由参数化**：候选只是参数化建议——
pin 常量收窄 range 都是合法走法（词表外动作/未知参数会被拒，词表内自由）。

决策启发：reward 领先且有余量 → 深挖该候选；新候选未探 → 广度优先；某候选 u_inner
长期垫底 → 预算不再给它；board 平台期 → 先 preview 再决定收局。

## 控制面

- `modeling_rollback`：回退 models/board 到上一快照（预算不回补）——run 后想撤销
  最近一次账本变动时用；
- `modeling_finish(preview=true)`：**收局前标准动作**——零副作用看 board/搜索证据/
  预算，不碰 holdout；
- `modeling_finish(deliver_keys=...)`：真收局——每交付模型 holdout 恰读一次 →
  terminal_utility + findings + **rank_consistency**（SH 赌注出账：u_inner 与 u_full
  的秩相关——低秩相关说明 partial 截断失真，加重跑深度）；缺省交付全部已训练候选；
  accepted 恒真（test 读即终局，无拒收重试）；重入 = 幂等重导出；
- `modeling_abandon(reason=...)`：用户中止/改协议重开前显式终局（不留僵尸）。

## 纪律

- **no-peeking**：run 全程不碰 holdout；board 只存 u_inner；不以"想象中的 test 分数"
  做任何决策；
- findings 是回流的载体：误差集中列 → 建议用户回清洗域修列（建议式，不自动改）；
- rank_consistency 是 SH 的诚信账——每次结算看一眼，partial 序不可信时调策略；
- 应用（predict/inverse_optimization）不在本 MDP——loader 引用（sid 相对寻址）是
  finish 交付物，下游工具按它寻址。

## 参考

- 引擎与两段式：`packages/model/mflowy/model/session.py`
- 设计依据：Jamieson & Talwalkar 2016（Successive Halving）+ Li et al. 2016（Hyperband）
- 快照（取代前的相位机诊断）：docs/research/2026-10-09-modeling-mdp-design.md
