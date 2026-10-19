---
name: model
description: 建模训练 MDP 回合操作手册(mflowy)。CVCard 编译后开始建模回合，或建模会话进行中(收到 session_id)时使用。配合 mflowy MCP 工具 modeling_session_start / modeling_propose / modeling_apply(action) / modeling_rollback / modeling_report_findings / modeling_finish / modeling_abandon 驱动回合循环——动作空间 = 八动作面（配置编辑 + 计算动作），环境面在引擎，π(策略)是你。前置:model-brief skill 已产出 CVCard（含 research-review 结构化候选）。
---

# Model Skill：建模 MDP 回合循环（search-only 回路 + 先搜后店）

三段式研究流第三段。你的职责是**策略（π）**：读账本 board（u_inner 记分牌 + findings
诊断面）、决定光标指向哪个候选、怎么参数化、何时搜索/终训/收局。环境面（增量 study/
预算/no-peeking 守卫/决策日志）在会话引擎。**holdout 全程只在 final_train_test 被读
（每 combo 恰一次，读完即冻结）**——回路内你看到的永远只是 inner/OOF 证据。

## 开局

```
modeling_session_start(final_path, card=<DomainCard>, cv_card=<CVCard>)   # 唯一主通道
```

- 协议由 **model-brief skill**（含 research-review 结构化候选）产出——candidates =
  `[{family, baseline, space_hints, seed}]`（baseline 恰一；hints 是缺省 pins/narrowing，
  你回合内可覆盖——卡是先验不是边界）；start 即 D 物化冻结 + 单一预算起算；换协议 =
  新 episode（弃局重开）。

## 回合：八动作面（`modeling_apply(action, params)`）

配置编辑（零步费零奖励，只改光标，生效于下一次 run_hpo）：
`select_model{family}` 定候选（注入该候选 hints/seed 缺省）→
`set_hp{param: 值}` pin 常量 / `set_hp_space{param: [lo,hi]|[v1,v2,…]}` 收窄动态维 /
`set_seed{seed}`（换 seed = 换目标 = study 重开）→ `select_FE{node}` 换 FE 档
（对当前候选结构非冗余的槽才合法）。

计算动作（吃步；预算 = 实验数）：
1. `run_hpo{n_trials}`——**search-only**：sqlite study 增量续跑（空间收窄/pin/加深
   继续，seed/scoring 变重开）；每 trial inner CV → u_inner/std；结束在 θ* 上出 OOF
   预测 + 图集 + findings（diagnose 四算子：error_concentrated/slice_collapse/
   fold_std/complexity_edge）；响应 `reward` = **Δu_inner**（相对该键上次最优）。
2. `final_train_test`——**先搜后店**：θ* 从账本读 → 全量拟合 → holdout 终评恰一次 →
   产物挂 run → **combo 冻结**（同键不可再搜索）；reward = 0（u_full 只进 verdict 面）。

`modeling_propose` 菜单 = findings 驱动层（记分牌序浮于顶部：折间不稳→加深、误差
集中→FE 换档、贴边界/过拟合→正则收窄、切片塌陷→类别权重）+ 基座层（baseline 第一、
未跑候选、终训交付、配置动作面）。规则 findings 与你落诊的 LLM findings 均匀聚合。

**预算即决策**：只有计算动作吃步——配置编辑自由试错；好钢（run_hpo 深度）用在 board
最有希望的方向。**自由参数化**：菜单只是建议不是边界（词表外动作/未知参数会被拒）。

## 控制面

- `modeling_rollback`：回退**光标**到上一快照（预算不回补；账本与盘上 study 永不
  回退——append-only 审计面）；
- `modeling_report_findings(key, findings)`：人读诊断落账本（kind ⊆ 词表、severity
  ∈ [0,1]；决策日志审计）——规则算子覆盖不了的判断走这里；
- `modeling_finish(preview=true)`：**收局前标准动作**——聚合全部已训 combo 的产物
  清单（OOF 图集/预测长表/指标）+ 绘泰勒（holdout 终训预测 fan-in）→ 你用宿主
  多模态读图分析；
- `modeling_finish(conclusion=...)`：真收局——verdict（u_full/findings/rank_consistency）
  + **LaTeX PDF 报告**（逐模型分析/泰勒/结论/rank）进交付面；conclusion 写「为什么
  是这个模型」（findings 装「哪里有问题」——语义不混）；重入 = 幂等重导出
  （conclusion 不可变）；
- `modeling_abandon(reason=...)`：用户中止/改协议重开前显式终局（不留僵尸）。

## 纪律

- **no-peeking**：回路内不碰 holdout；board 只存 u_inner；不以"想象中的 test 分数"
  做任何决策；final_train_test 读完即冻结——不可用 test 反复挑参；
- findings 是回流的载体：误差集中列 → 建议用户回清洗域修列（建议式，不自动改）；
- rank_consistency 是 SH 的诚信账——每次结算看一眼，inner 序不可信时调策略；
- 应用（predict/inverse_optimization）不在本 MDP——loader 引用（sid 相对寻址）是
  finish 交付物，下游工具按它寻址。

## 参考

- 引擎与状态机：`packages/builtin_plugins/model/mflowy/model/session.py`（八动作面/
  增量 study/freeze）；诊断算子：`mflowy/model/diagnose.py`；出题规则：
  `mflowy/model/propose_rules.py`
- 设计依据：Jamieson & Talwalkar 2016（SH）——Δu_inner 奖励即 diminishing-return 信号
- 快照（取代前的回路设计）：docs/research/2026-10-09-modeling-mdp-design.md
