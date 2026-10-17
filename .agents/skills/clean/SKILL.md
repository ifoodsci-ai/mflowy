---
name: clean
description: 数据清洗 MDP 回合操作手册(mflowy)。用户要求"开始清洗/自主清洗/修数据"，或清洗会话进行中(收到 session_id)时使用。配合 mflowy MCP 工具 clean_session_start / clean_propose / clean_apply / clean_rollback / clean_finish / clean_abandon 驱动回合循环——环境面在引擎，π(策略)是你。前置:survey skill 已产出 DomainCard。
---

# Clean Skill：清洗 MDP 回合循环

三段式研究流第二段：`建档与 EDA →(冻结)→ 清洗 MDP →(冻结)→ 建模 MDP`。
你的职责是**策略（π）**：读观测、选动作、必要时回滚、决定收局。环境面（转移/
观测/奖励/预算/守卫）在会话引擎——你不重算状态，引擎每步免费给你 s′ 观测。

## 开局

```
clean_session_start(file_path, card=<DomainCard>, max_steps=15)   # 唯一主通道，直进动作面
```

- 卡由 **survey skill**（warm-up/dive-in 问卷剧本）产出后传入——start 即全量校验
  （列引用/物理范围 fail-loud）并富化 never_drop；never_drop / 目标列禁改 /
  unit_hints / expected_ranges / defect_priority / 保留率下限全部生效；
- 返回 s′：缺陷清单（kind×count）+ 内生质量分 + 预算。改卡 = 新 episode（重开会话）。

## 回合循环

1. `clean_propose(session_id)` —— 合法走法生成器：错误类型 → 参数化候选 + rationale。
   **菜单是建议不是边界**：词表外（clean 目录之外）的动作会被拒，**词表内的参数
   可自由调**——候选只是参数化建议（带证据的默认档），按领域判断改参不必照抄。
2. 拿不准先 `clean_apply(..., dry_run=true)` 预览 diff（200 行样本，不占步数）。
3. `clean_apply(session_id, node, params)` 提交：守卫 → 转移 → 免费观测
   （diff + 缺陷清单）+ 决策日志 (s,a,r,s′)。响应自带 **`reward`**（本步 ΔΦ，
   舍入 4 位）与 **`state_digest`**（缺陷摘要指纹）——直接读数，不从日志做减法。
4. 读 s′ 决策：
   - reward > 0、缺陷减 → 继续 ①；
   - reward < 0 / 引入新缺陷 / 误判 → `clean_rollback(session_id, to_version)`（版本树回退，
     决策日志保留；预算不回补；响应 reward = 恢复 ΔΦ）；
   - 错误清单清空 / 预算耗尽 / 连续多步 reward≈0（平台期）→ 先 preview 再决定收局。

## 收局

```
clean_finish(session_id, min_quality=<可选 θ 门>, preview=<可选只读探针>)
```

- **preview 流**（平台期标准动作）：先 `preview=true` 看 `terminal_utility.skill`
  ——零副作用（不收局、不导出、θ 门不生效），会话原样可继续；
- `terminal_utility` 是顶层键（拒收/accepted 两路一致）：skill ≤ 0 =**「干净但
  不可学」警戒线**→ 回滚撤过度删改换温和档位（preview 使该建议首次可执行），
  skill > 0 且平台 → 真收局；小数据下单次 CV 有方差，当线索不当判决；
- θ 门：`min_quality` 设阈值时质量不达标 → `accepted=false` 返回继续（会话保留），
  此时**不要**机械重试同一动作——换走法或回滚；
- 拒收后重收局：不带 `min_quality` 再 finish 即导出（人为你确认终态）；
- **响应丢失自愈**：已收局的会话重调 finish = 幂等重导出同一交付物（已弃局则不可）；
- 真收局返回：`terminal_utility` + `final_quality` + `plan_yaml`（可执行清洗计划）
  + 终态数据路径 + 全程决策日志。

## 弃局

```
clean_abandon(session_id, reason=<可选，落盘供审计>)
```

- 用户中止清洗 / 改卡重开新 episode 前，显式终局旧会话——**不留僵尸**；
- 决策日志与版本树保留在盘（审计面完整），但不导出交付物；已终态再弃即报错。

## 纪律

- **生命周期纪律**：要么 finish 要么 abandon——回合结束必居其一，不留僵尸；
- **续局**：上下文死亡/新实例接手 = 拿 `session_id` 直接 `clean_propose` 续跑
  （响应自带 card/预算/步数，无需额外查询）；
- **预算观感**：max_steps 是硬预算，rollback 不回补——好钢用在 reward 最大的动作上；
- 每步 rationale 从 propose 带出，收局后 decision_log 是全程审计面（可重放）；
- 你不做检测（观测免费自带）、不越词表（clean 目录外的动作会被拒）、不改 DomainCard。

## 参考

- 引擎与守卫：`packages/data/mflowy/data/cleaning_session.py`
- 设计快照（会话版五元组映射）：docs/research/2026-10-09-cleaning-mdp-design.md §7
