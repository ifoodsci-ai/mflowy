# mflowy-data（数据域）

MFlowy 数据域包：清洗 MDP 的动作域——load / statistic / detect / clean 四动作族 +
plot EDA 图族（族内扁平，直住 `plot/`）。数据栈为硬依赖，装包即得完整数据域（词表 = 已安装能力）。

- 动作族：`load/`（6 节点）、`statistic/`（profile=info 薄封装 / univariate 单变量 typed 统计+规则化 findings / bivariate 双变量三路无 target 全对——效应量定档 p 把门）、`detect/`（5 缺陷算子 +
  ErrorReport 契约 + quality_score + `detect/rules.py` 共享规则——与 clean 原子同尺的
  唯一居所）、`clean/`（19 命名原子，族内扁平六模块：drop×6 / fill×7（ffill/bfill 带 by= 组内
  防泄漏）/ cap×3（cap_iqr·cap_z·range_clip）/ add_missing_flag / strip_units / python——
  策略在名、级联删除、类型族不适配即报错；`.repair` v2 全员候选 + 记分牌排序，仅条件型
  python/range_clip 不挂）
- 共享件：主题门面上住 `mflowy-utils[plot]`（`mflowy.utils.plot`——Agg 后端 + seaborn
  主题就绪 + 色板/字体常量；本包 `plot/__init__` 薄 re-export，模型域评估图同源复用）、
  `middlewares/`（数据面遥测 + df_columns 诊断）
- 类型：`types.py`（TASKTYPE 任务类型——target 列的数据事实；DomainCard 领域建档卡）
- 规则聚合：`repair_rules.py`（cleanRuleEngine 默认实例——词表 `.repair` 声明包装注册为
  RepairRule；propose 走 `rule_engine_rwa` → fire/poll，会话实例 pkl 持载自定义规则）
- 判据：镜像路径有对应测试（`tests/`），`uv sync` 后 `list-nodes` 可见（新模块须重跑 `uv sync`）
- 第三方插件参考：以 mflowy 为 base 依赖，声明 `[project.entry-points."mflowy.plugins"]` 即成插件
