---
name: model
description: 建模训练 POMDP 回合操作手册(mflowy)。开局产物就绪后开始建模回合，或建模会话进行中(收到 session_id)时使用。配合 mflowy MCP 工具 modeling_session_start / modeling_observe / modeling_apply(action) / modeling_rollback / modeling_report_findings / modeling_finish / modeling_abandon 驱动回合循环——动作空间 = 八动作面（配置编辑 + 计算动作），环境面在引擎，π(策略)是你。会话恒单目标任务——多目标在会话外三层循环（多目标→多模型→单目标任务会话，每目标各开一会话）。前置:model-brief skill 已产出开局入参（cv_card = cv.split 配置 yaml 明文 + 显式协议参数，含 research-review 结构化候选）。
disable-model-invocation: true
---

# Model Skill：建模 POMDP 回合循环（search-only 回路 + 先搜后店）

三段式研究流第三段。你的职责是**策略（π）**：读账本 board（u_inner 分数面 + findings
诊断面）、决定光标指向哪个候选、怎么参数化、何时搜索/终训/收局。环境面（增量 study/
预算/no-peeking 守卫/决策日志）在会话引擎。**holdout 全程只在 final_train_test 被读
（每 combo 恰一次，读完即冻结）**——回路内你看到的永远只是 inner/OOF 证据。

## 开局

```
modeling_session_start(final_path, cv_card=<cv.split 配置 yaml 明文>,
                        target=, task=, metric=, candidates=, max_steps=, id_columns=)   # 唯一主通道：显式参数 > card(DomainCard) 背景分层缺省
```

- 协议由 **model-brief skill**（含 research-review 结构化候选）产出——candidates =
  `[{family, baseline, space_hints, seed}]`（baseline 恰一；hints 是缺省 pins/narrowing，
  你回合内可覆盖——卡是先验不是边界）；start 即 D 物化冻结 + 单一预算起算；换协议 =
  新 episode（弃局重开）。

## 回合：八动作面（`modeling_apply(action, params)`）

配置编辑（零步费零奖励，只改光标，生效于下一次 run_hpo；**单表 H**：同一参数
唯一槽位——pin 覆盖空间或反向都是显式 last-write-wins，决策日志记前后值）：
`select_model{family}` 定候选（注入该候选 hints/seed 缺省）→
`set_hp{param: 值}` pin 常量 / `set_hp_space{param: [lo,hi]|[v1,v2,…]}` 收窄动态维 /
`set_seed{seed}`（换 seed = 换目标 = study 重开）→ `select_FE{node}` 换 FE 档
（对当前候选结构非冗余的槽才合法）。

计算动作（吃步；预算 = 实验数）：
1. `run_hpo{n_trials}`——**search-only**：sqlite study 增量续跑（空间收窄/pin/加深
   继续，seed/scoring 变重开）；每 trial inner CV → u_inner/std；结束在 θ* 上出 OOF
   预测 + 图集 + findings（diagnose 四算子：error_concentrated/slice_collapse/
   fold_std/complexity_edge）；响应 `reward` = **Δu_inner**（相对该键上次最优）。
2. `final_train_test`——**先搜后店**：终训走 modeling 工具（JobProvider DAG）——
   交付的恒是**搜索时刻证据配置**（快照 ∪ θ*，搜后改 pin 不生效）→ 终训/终评/
   X_test·y_test·y_pred 副产物/plan 一体（交付的 plan_yaml 就是那次终训本身）→ holdout
   终评恰一次 → **combo 冻结**（同键不可再搜索）；reward = 0（终局效用 u 只进 verdict 面）。

**读观测，自构动作**——引擎不出菜单：`modeling_observe(session_id)`（幂等只读）回
决策面——`findings`（**只投影本 combo 键**，证据不跨 FE 组合；规则 findings 与你落诊
的 LLM findings 均匀聚合）、`combo_key`、`models` 账本摘要（θ*/u_inner/冻结旗）、
`has_cat`、`face`（各候选族超参面：动态维可收窄、静态常量可 pin）。

决策指南（起点不是边界）：
- **推进序**：baseline 候选先跑（定参照系）→ 未跑候选 `select_model` + `run_hpo`
  首搜（n_trials=10）→ 已搜 combo `final_train_test` 交付（holdout 恰一次）；
- **findings → 动作**：`fold_std`（折间不稳）→ `run_hpo` 加深（n_trials ≈ max(8,
  3×上次)）；过拟合族（`overfit_train_val_gap`/`overfit_val_loss_rise`/`drift`/
  `complexity_edge`）→ 族正则参数 `set_hp_space` 收窄到上半区（静态面则 `set_hp` pin）；
  欠拟合族（`underfit_high_close`/`underfit_loss_plateau`）→ `select_FE` 换档加信号 +
  正则放宽到下半区；`error_concentrated` → `select_FE` 换编码/缩放重跑；
  `slice_collapse` → 族有 `scale_pos_weight` 时 `set_hp` 提少数类权重（如 2.0）；
- **收局**：预算将尽 / board 平台期 / 交付面已足 → 走收局剧本（停止没有引擎提示）。

**预算即决策**：只有计算动作吃步——配置编辑自由试错；好钢（run_hpo 深度）用在 board
最有希望的方向。**自由参数化**：词表内动作与参数自由组合（词表外动作/未知参数会被拒）。

## 控制面

- `modeling_rollback`：回退**光标**到上一快照（预算不回补；账本与盘上 study 永不
  回退——append-only 审计面）；
- `modeling_report_findings(key, findings)`：人读诊断落账本（kind ⊆ 词表、severity
  ∈ [0,1]；决策日志审计）——规则算子覆盖不了的判断走这里；
- **收局剧本固定序**（素材两工具 → 撰写 → 收编）：
  1. `draw_taylor_diagram(session_id, run_ids)`：多终训 run 副产物画泰勒图（误差结构
     对比——终局排名的证据面；各模型 y_test 不一致会拒绝出图）；
  2. `modeling_report_skeleton(session_id)`：报告素材样板——表格数字从账本渲染
     （u_inner/θ*/u/rank 抄文件非默写）+ 图片清单（OOF 图集/终训副产物/泰勒图路径）
     + 叙事槽（「为什么是这个模型」）——宿主多模态读图后撰写终局报告；
  3. `modeling_finish(session_id, pdf=...)`：**单动作收局**——引擎收编你的报告文件进
     deliverables（原子落盘；不编译不验内容——数字权威面是 verdict/board，报告是给
     人看的）+ verdict（u/findings/rank_consistency）；**pdf 必须在报告就绪后
     调用**（文件缺失 fail-loud）；重入 = 幂等重导出（交付报告不可变更——重写须弃局
     重开）；
- `modeling_abandon(reason=...)`：用户中止/改协议重开前显式终局（不留僵尸）。

## 纪律

- **no-peeking**：回路内不碰 holdout；board 只存 u_inner；不以"想象中的 test 分数"
  做任何决策；final_train_test 读完即冻结——不可用 test 反复挑参；
- findings 是回流的载体：误差集中列 → 建议用户回清洗域修列（建议式，不自动改）；
- rank_consistency 是 SH 的诚信账——每次结算看一眼，inner 序不可信时调策略；
- 应用（predict/inverse_optimization）不在本 POMDP——loader 引用（sid 相对寻址）是
  finish 交付物，下游工具按它寻址。

## 参考

- 设计依据：Jamieson & Talwalkar 2016（SH）——Δu_inner 奖励即 diminishing-return 信号
- 快照（取代前的回路设计）：docs/research/2026-10-09-modeling-mdp-design.md
