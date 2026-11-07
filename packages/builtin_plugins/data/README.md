# mflowy-data（数据域）

MFlowy 数据域包：清洗 POMDP 的动作域——load / statistic / detect / clean 四动作族 +
plot EDA 图族（族内扁平，直住 `plot/`）。数据栈为硬依赖，装包即得完整数据域（词表 = 已安装能力）。

## 清洗会话 POMDP 元素表（O/S/A/P·T/R）

契约下限与校验器住 `driver/pomdp.py`（Observation / FinishVerdict / Candidate）；本域语义如下（引擎 = `cleaning_session.py`。转移确定性——无概率
转移表；可观测性 = 检测算子覆盖面：O 是检测算子族的确定性多对一投影，覆盖外缺陷对 π 不可见
（态混叠），终局 U（可学性）依赖观测未含的态——「干净但不可学」即其症状）：

| 元素          | 本域定义                                                                                   | 居所                                   |
| ------------- | ------------------------------------------------------------------------------------------ | -------------------------------------- |
| O 观测        | 检测算子族即观测函数（确定、多对一）——簿记态全可见，真态可见性以算子覆盖面为界；s′ 随 apply 免费自带（错误清单 + Φ + diff）；同（指纹 × 卡先验）memo 防重算 | `_detect_all` / `_state_response`      |
| S 状态        | 真态 = 数据本体（当前版本 parquet 内容）+ 冻结卡 + 预算 + 决策日志；`state_digest`（defects/quality/当前版本指针）是其检测面投影，非状态本体 | `_digest` / `decision.log` 单记录版本树        |
| A 动作        | 运行期 clean 词表（词表 = 已安装能力，见上文动作族）；终态动作面为空                       | `_allowed_actions` / `apply_action`    |
| P·T 转移      | 守卫（never_drop 存活 / 目标列值禁改 / 保留率下限）→ 单步流编译执行（Conf→Builder→DAG，#89 ②片——apply 与导出计划同构）→ 新 parquet 版本 + 日志追加（commit 点）；rollback = 记录 + 指针回拨（append-only，弃分支盘上可回，#89 ④片） | `_guard` / `apply_action` / `rollback` |
| R 奖励        | 过程奖励 = ΔΦ（前后同尺、卡权重双侧生效）；rollback 奖励 = 恢复的 ΔΦ；终局 U = train-fold 快速学习器 skill（只在 finish 出现，θ 验收门可拒收） | `apply_action` / `_terminal_utility` / `finish_session` |

- 动作族：`load/`（6 节点）、`statistic/`（profile=info 薄封装 / univariate 单变量 typed 统计+规则化 findings / bivariate 双变量三路无 target 全对——效应量定档 p 把门）、`detect/`（8 缺陷算子 +
  DefectType/Severity 封闭枚举 + Finding 六字段契约 + quality_score + `detect/rules.py`
  共享判据（zscore/iqr 栅栏、脱敏、suspected_id 打分谓词）、`clean/`（23 命名原子：drop×6 /
  fill×7（ffill/bfill 带 by= 组内防泄漏）/ cap×3 / add_missing_flag / strip_units /
  mask_cols·hash_cols（匿名化）/ cast_cols·to_category（类型档）/ python——
  策略在名、级联删除、类型族不适配即报错；选项面 = 词表全集（π 自构动作，菜单随 #113 退役；
  条件型 python 无对应缺陷类型））
- 共享件：主题门面上住 `mflowy-utils[plot]`（`mflowy.utils.plot`——Agg 后端 + seaborn
  主题就绪 + 色板/字体常量；本包 `plot/__init__` 薄 re-export，模型域评估图同源复用）、
  `middlewares/`（数据面遥测 + df_columns 诊断）
- 类型：`types.py`（TASKTYPE 任务类型——target 列的数据事实；DomainCard 领域建档卡）
- 观测面：`clean_observe` 幂等只读（续局锚点）——候选规则与菜单随 #113 退役，缺陷→算子的
  策略知识外置 clean skill（决策指南）
- 判据：镜像路径有对应测试（`tests/`），`uv sync` 后 `list-nodes` 可见（新模块须重跑 `uv sync`）

## 存活不变量对照（测试名自述 ↔ 设计文档 INV 编号，#119）

不变量不设编号面（AGENTS/CONTEXT 零编号）——存活不变量以测试名自述，设计文档 §11 的
INV 编号在此对照存档；已 dissolve 项随载体死（INV-3 批量提交防自相矛盾、INV-7 空 cols
静默容忍、INV-9 批量化排序、dry_run——#113/#118/#119 裁定）：

| 不变量 | 自述测试 | 居所 |
| --- | --- | --- |
| INV-1 rationale score 前缀 | `test_rationale_requires_score_prefix` / `test_smoke_every_finding_holds_contract` | `tests/detect/test_errors.py`、`test_smoke.py` |
| INV-2 evidence 脱敏 | `test_evidence_samples_are_masked` / `test_mask_text` | `tests/detect/` |
| INV-4 转形：finish 残留全带 keep 理由 | `test_finish_keep_declarations_residual_manifest` | `tests/test_cleaning_session.py` |
| INV-5 转形：intent 派生 | `test_apply_decision_surface_finding_ids_and_reason` | 同上 |
| INV-6 clean 原子返回新 df | `test_clean_op_runs_with_min_params_returns_new_df` | `tests/clean/test_vocab_smoke.py` |
| INV-8 clean_op ∈ 词表 + params 合 schema | `test_apply_rejects_non_clean_node` / `test_registry_matches_tools_module` | 会话与工具镜像测试 |
| INV-10 转形：算子不暴露为 π 工具 | `test_menu_tool_retired_observe_in`（API 面事实） | `tests/test_data_tools_registry.py` |
- 第三方插件参考：以 mflowy 为 base 依赖，声明 `[project.entry-points."mflowy.plugins"]` 即成插件
