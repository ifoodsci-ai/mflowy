---
name: research-review
description: 文献候选评审(mflowy)。model-brief 问卷前使用——用户确认开始建模且需要文献依据选模型候选时触发。相似/跨领域研究问题 → 结构化模型候选清单（candidates：baseline 锚点 + space_hints + seed），经 CVCard 进建模 MDP。前置:clean skill 已 finish。
---

# Research-Review Skill：文献候选评审（剧本层，接 model-brief 之前）

职能：把"选哪些模型族"的第一轮淘汰与搜索起点交给**文献先验**而非暴力筛选。产物只有
`candidates` 一个字段，交给 model-brief 编译进 CVCard——本 skill 不碰数据、不碰引擎、
不产出模型。

## 相似研究问题 → 结构化模型候选

1. 从 DomainCard（domain/goal/target 语义）+ 清洗收局报告提炼**研究问题画像**：
   数据规模 / 特征形态（表格·宽窄·类别占比）/ 任务类型 / 量级约束；
2. 检索同领域与跨领域的相似问题基准与方法（Kaggle 同型赛题、领域 survey、
   仓库 examples/）；
3. 产出**结构化候选清单** `[{family, baseline, space_hints, seed}]`（对齐运行期族
   目录——已声明 `.mdp_family` 者；清单外族 = 第一轮淘汰）：
   - **baseline 恰一**：最稳的文献锚点候选（propose 时位于第一——先跑它定参照系）；
   - 每候选一句文献依据（"该规模表格基准上 GBDT 系长期占榜"）；
   - **space_hints**（可选）：文献知识注入搜索起点——标量 = pin 缺省（如
     `n_estimators: 500`）、区间 = 收窄（如 `max_depth: [4, 8]`）；保持缺省则留空；
   - **seed**（可选）：候选默认随机种子（搜索期内冻结——换 seed = 换目标 = study 重开）；
   - 无证据支撑的族不入清单（宁缺勿编——留空则全部已声明族开放）。

## 收口：交给 model-brief

产物随必答项一起进 CVCard：`{candidates: [结构化候选清单]}`——开局校验（候选 ⊆
已声明目录、baseline 恰一、hints ⊆ 族签名声明空间）fail-loud，坏值原样反馈修正。

## 纪律

- 本 skill 的全部产出是一个字段——不做特征工程、不做超参决策（hints 是**起点先验**，
  π 回合内可覆盖——卡是先验不是边界）；
- 文献依据可追溯：每候选一条出处或机理陈述，用户可否决；
- 候选清单是**开局边界**：清单外族被引擎拒绝（守卫拦"事实不成立"）——想跑某族
  就应在清单里声明它；
- 搜索深浅（n_trials）不经本 skill——π 在回合里按空间摘要与 findings 自行分配。

## 参考

- 消费方：model-brief skill（编译进 CVCard）→ modeling_session_start
- 族目录声明：`packages/builtin_plugins/model/mflowy/model/registry.py`
  （`.mdp_family` = 目录标记 + structure + machinery）
