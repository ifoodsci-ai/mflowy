---
name: clean
description: 数据清洗 POMDP 回合操作手册(mflowy)。用户要求"开始清洗/自主清洗/修数据"，或清洗会话进行中(收到 session_id)时使用。配合 mflowy MCP 工具 clean_session_start / clean_observe / clean_apply / clean_rollback / clean_finish / clean_abandon 驱动回合循环——环境面在引擎，π(策略)是你。前置:survey skill 已产出 DomainCard。
disable-model-invocation: true
---

# Clean Skill：清洗 POMDP 回合循环

三段式研究流第二段：`建档与 EDA →(冻结)→ 清洗 POMDP →(冻结)→ 建模 POMDP`。
你的职责是**策略（π）**：读观测、选动作、必要时回滚、决定收局。环境面（转移/
观测/奖励/预算/守卫）在会话引擎——你不重算状态，引擎每步免费给你 s′ 观测。

## 开局

```
clean_session_start(file_path, card=<DomainCard>, max_steps=15)   # 唯一主通道，直进动作面
```

- 卡由 **survey skill**（warm-up/dive-in 问卷剧本）产出后传入——start 即全量校验
  （列引用/物理范围 fail-loud）并富化 never_drop；never_drop / 目标列禁改 /
  unit_hints / expected_ranges / defect_priority / 保留率下限全部生效；
- 返回 s′：缺陷清单（`state_digest.defects` 计数 + `problems` 逐条证据）+ 内生
  质量分 + 列画像（`columns`：dtype 族 / 分布读数）+ 预算。改卡 = 新 episode（重开会话）。

## 回合循环

1. **读观测，自构动作**——引擎不出菜单：可执行算子 = clean 词表全集
   （`list_nodes(namespace="clean")` 看全集、`get_node_info` 看参数面）；缺陷证据
   在 `state_digest.problems`，列读数在 `state_digest.columns`。按下方决策指南
   选算子与参数——指南是起点不是边界，词表内的算子与参数可自由组合（词表外的
   动作会被拒）。
2. `clean_apply(session_id, node, params, finding_ids, reason)` 提交：守卫 → 转移 →
   免费观测（失败即免费异常——不耗预算、不落版本、不进决策日志；试探不确定动作
   就直接提交，负 reward 即回滚，全量真值决策）+ 决策日志 (s,a,r,s′)。响应自带
   **`reward`**（本步 ΔΦ，舍入 4 位）与 **`state_digest`**——直接读数，不从日志做减法。
   - **`finding_ids`**（必填可空）：本步处理的 finding_id（`problems` 的
     `finding_id`）——非空时校验 ⊆ 当前 findings，幻觉 id 免费失败；行级算子
     （drop_duplicate 等）/逃生舱可传空；
   - **`reason`**（必填非空）：本步决策理由——每步可审计；
   - no-op 否决：不动 df 的提交被拒——保留不动 df 的处置走 `clean_finish` 的
     keep 声明。
3. 读 s′ 决策：
   - reward > 0、缺陷减 → 继续 ①；
   - reward < 0 / 引入新缺陷 / 误判 → `clean_rollback(session_id, to_version)`（版本树回退，
     决策日志保留；预算不回补；响应 reward = 恢复 ΔΦ）；
   - 缺陷清单清空 / 预算耗尽 / 连续多步 reward≈0（平台期）→ 先 preview 再决定收局
     （停止没有引擎提示——收局时机由你从观测判断）。

## 决策指南（缺陷 → 算子 × 参数档位）

| 缺陷（defect_type） | 温和档 | 激进档 / 备注 |
| --- | --- | --- |
| high_null（高空值列，报告线 5%） | 数值列：偏斜（`columns` 的 \|skew\| > 1）→ `fill_median`，近对称 → `fill_mean`，有序趋势 → `fill_interpolate`；类别列 → `fill_mode` 或 `fill_const`(value="Unknown"，保留缺失信号)；时间列 → `fill_ffill`/`fill_bfill`（`by=` 组内防泄漏） | 空值率 ≥ 0.3 先 `add_missing_flag` 固化缺失信号（投资动作：Φ 不涨、回报在终局 U）；≥ 0.8 → `drop_cols`；行级清理 → `drop_missing_rows`(cols)。低空值列不产缺陷但 `columns` 画像 missing_rate 常驻——可自行决策动手 |
| empty_row（全空行） | `drop_missing_rows`(how="all")——文件噪声，零信息 | — |
| duplicate（重复行） | `drop_duplicate`(keep="first") | 重复即不可信 → keep="none" |
| outlier_numeric（数值离群） | `cap_iqr`(k=1.5，偏斜/重尾稳健)、`cap_z`(k=3，近正态) | `drop_outlier_iqr` / `drop_outlier_z`（行删除，看保留率） |
| collinear（共线对） | `drop_cols`：每对删方差小者（`columns` 的 var），卡认领列不删 | — |
| constant_col（常量列） | `drop_cols`（零信息） | — |
| format_anomaly（格式异常） | `strip_units`(columns={列: 单位})——单位优先取卡 unit_hints，缺则读证据的众数后缀 | 纯数值字符串 `cast_cols`(target_dtype="float")；类别声明 `to_category` |
| suspected_id（疑似标识列） | 确认后 `drop_cols` 出特征 | 需保留可连接性 → `hash_cols`（假名化，同值同哈希）/ `mask_cols`（脱敏） |
| suspected_composite_id（疑似组合键） | 确认后 `drop_cols` 整组出特征 | join 语义保留原列，finish 时 keep 声明 |
| prior_conflict（先验与数据矛盾） | 处置收敛为 **keep**——finish 时带理由声明（cite card.source 权威度；manual 源建议复核先验） | — |
| range_violation（物理越界） | `range_clip`(ranges=卡 expected_ranges) | `drop_outside`(cols, lo, hi) 删越界行 |

观测常驻 `routing_hints`（defect_type → 推荐算子清单，提示非约束）；残留缺陷可
带理由 keep 声明收局（`clean_finish` 的 `keeps`，不动 df、不吃步数）。

**反模式**：对目标列填充/截断（守卫拒——行删除允许）；一步删掉大量行（保留率下限拒）；
reward ≤ 0 后同方向加码；只追 Φ 忽略可学性（见收局的 skill 警戒线）。

## 收局

```
clean_finish(session_id, keeps=<可选 keep 声明>, min_quality=<可选 θ 门>, preview=<可选只读探针>)
```

- `keeps = [{finding_id, reason}]`：残留缺陷的保留理由（逐条声明——收局审计面
  `residual` 附理由、`manifest` 汇总 ID 先验与 keep 面供下游建模参考）；幻觉
  finding_id 会被拒。

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
- **续局**：上下文死亡/新实例接手 = 拿 `session_id` 直接 `clean_observe(session_id)`
  （幂等只读：响应自带 card/预算/步数/缺陷清单，无需额外查询）；
- **预算观感**：max_steps 是硬预算，rollback 不回补——好钢用在 reward 最大的动作上；
- 收局后 decision_log 是全程审计面（可重放）；
- 你不做检测（观测免费自带）、不越词表（clean 目录外的动作会被拒）、不改 DomainCard。

## 参考

- 设计快照（会话版五元组映射）：docs/research/2026-10-09-cleaning-mdp-design.md §7
