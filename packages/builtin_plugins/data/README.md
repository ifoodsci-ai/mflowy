# mflowy-data（数据域）

MFlowy 数据域包：清洗 POMDP 的动作域——load / statistic / detect / clean 四动作族 +
plot EDA 图族（族内扁平，直住 `plot/`）。数据栈为硬依赖，装包即得完整数据域（词表 = 已安装能力）。

## 清洗会话 POMDP 元素表（O/S/A/P·T/R）

契约下限与校验器住 `driver/mdp_session.py`（Observation / FinishVerdict / Candidate——模块名
为代码锚点，不随术语改名）；本域语义如下（引擎 = `cleaning_session.py`。转移确定性——无概率
转移表；可观测性 = 检测算子覆盖面：O 是检测算子族的确定性多对一投影，覆盖外缺陷对 π 不可见
（态混叠），终局 U（可学性）依赖观测未含的态——「干净但不可学」即其症状）：

| 元素          | 本域定义                                                                                   | 居所                                   |
| ------------- | ------------------------------------------------------------------------------------------ | -------------------------------------- |
| O 观测        | 检测算子族即观测函数（确定、多对一）——簿记态全可见，真态可见性以算子覆盖面为界；s′ 随 apply 免费自带（错误清单 + Φ + diff）；同（指纹 × 卡先验）memo 防重算 | `_detect_all` / `_state_response`      |
| S 状态        | 真态 = 数据本体（当前版本 parquet 内容）+ 冻结卡 + 预算 + 决策日志；`state_digest`（kind×counts）是其检测面投影，非状态本体 | `_digest` / manifest `versions`        |
| A 动作        | 运行期 clean 词表（词表 = 已安装能力，见上文动作族）；终态动作面为空                       | `_allowed_actions` / `apply_action`    |
| P·T 转移      | 守卫（never_drop 存活 / 目标列值禁改 / 保留率下限）→ 原子执行 → 新 parquet 版本 + manifest commit；rollback 截断版本树（决策日志保留，审计不可篡改） | `_guard` / `apply_action` / `rollback` |
| R 奖励        | 过程奖励 = ΔΦ（前后同尺、卡权重双侧生效）；rollback 奖励 = 恢复的 ΔΦ；终局 U = train-fold 快速学习器 skill（只在 finish 出现，θ 验收门可拒收） | `apply_action` / `_terminal_utility` / `finish_session` |

- 动作族：`load/`（6 节点）、`statistic/`（profile=info 薄封装 / univariate 单变量 typed 统计+规则化 findings / bivariate 双变量三路无 target 全对——效应量定档 p 把门）、`detect/`（8 缺陷算子 +
  ErrorReport 契约 + quality_score + `detect/rules.py` 共享规则——与 clean 原子同尺的
  唯一居所）、`clean/`（19 命名原子，族内扁平六模块：drop×6 / fill×7（ffill/bfill 带 by= 组内
  防泄漏）/ cap×3（cap_iqr·cap_z·range_clip）/ add_missing_flag / strip_units / python——
  策略在名、级联删除、类型族不适配即报错；`.repair` v2 全员候选 + 记分牌排序，仅条件型
  python 不挂）
- 共享件：主题门面上住 `mflowy-utils[plot]`（`mflowy.utils.plot`——Agg 后端 + seaborn
  主题就绪 + 色板/字体常量；本包 `plot/__init__` 薄 re-export，模型域评估图同源复用）、
  `middlewares/`（数据面遥测 + df_columns 诊断）
- 类型：`types.py`（TASKTYPE 任务类型——target 列的数据事实；DomainCard 领域建档卡）
- 规则聚合：`repair_rules.py`（cleanRuleEngine 默认实例——词表 `.repair` 声明包装注册为
  RepairRule；propose 走 `rule_engine_rwa` → fire/poll，会话实例 pkl 持载自定义规则）
- 判据：镜像路径有对应测试（`tests/`），`uv sync` 后 `list-nodes` 可见（新模块须重跑 `uv sync`）
- 第三方插件参考：以 mflowy 为 base 依赖，声明 `[project.entry-points."mflowy.plugins"]` 即成插件
