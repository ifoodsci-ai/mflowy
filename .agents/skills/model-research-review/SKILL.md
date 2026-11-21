---
name: model-research-review
description: 文献候选评审(mflowy)。model-prior-survey 问卷前使用——用户开始建模前需要文献依据选模型候选时触发。相似/跨领域研究问题 → 结构化模型候选清单（candidates JSON——schema 一跳发现自 mflowy modeling_load_or_start -h），经 model-prior-survey 编入开局入参交棒 modeling 开局。前置:data-clean skill 已 finish。
disable-model-invocation: true
---

# Model-Research-Review Skill：文献候选评审（先验子剧本，接 model-prior-survey 之前）

职能：把"选哪些模型族"的第一轮淘汰与搜索起点交给**文献先验**而非暴力筛选。产物只有
`candidates` 一个字段（JSON），交给 model-prior-survey 编入开局入参——本 skill
不碰数据、不碰引擎、不产出模型、不开局。

## 相似研究问题 → 结构化模型候选

1. 从清洗先验（domain/goal/target 语义——data-prior-survey 产物）+ 清洗收局报告
   提炼**研究问题画像**：数据规模 / 特征形态（表格·宽窄·类别占比）/ 任务类型 /
   量级约束；
2. 检索同领域与跨领域的相似问题基准与方法（Kaggle 同型赛题、领域 survey、
   仓库 examples/）；
3. 产出**结构化候选清单**（对齐运行期族目录——已声明 `.mdp_family` 者；清单外族 =
   第一轮淘汰）。候选字段面**零硬编码**（#233）——一跳发现自
   `mflowy modeling_load_or_start -h` 的 `--candidates` 描述（校验闸同源：
   候选 ⊆ 已声明族目录、baseline 恰一、hints ⊆ 族签名声明空间）。评审要点：
   - **baseline 恰一**：最稳的文献锚点候选（modeling 主剧本推进序第一——先跑它
     定参照系）；
   - 每候选一句文献依据（"该规模表格基准上 GBDT 系长期占榜"）；
   - 文献知识注入搜索起点（标量 = pin 缺省、区间 = 收窄——字段语义见 --candidates
     描述）；保持缺省则留空；
   - 无证据支撑的族不入清单（宁缺勿编——留空则全部已声明族开放）。

## 收口：交给 model-prior-survey

产物 = `candidates` JSON——随必答项一起经 model-prior-survey 编入开局入参交棒
modeling；开局校验（候选 ⊆ 已声明目录、baseline 恰一、hints ⊆ 签名空间）
fail-loud，坏值原样反馈修正。

## 纪律

- 本 skill 的全部产出是一个字段——不做特征工程、不做超参决策（hints 是**起点先验**，
  π 回合内可覆盖——卡是先验不是边界）；
- 文献依据可追溯：每候选一条出处或机理陈述，用户可否决；
- 候选清单是**开局边界**：清单外族被引擎拒绝（守卫拦"事实不成立"）——想跑某族
  就应在清单里声明它；
- 搜索深浅（n_trials）不经本 skill——π 在回合里按空间摘要与 findings 自行分配。

## 参考

- 工具发现：记不清工具名或参数面时，先 `mflowy --help` 看分组清单、`mflowy <tool> --help` 看全参与用法（--candidates 嵌套字段面同此揭示）；
- 消费方：model-prior-survey skill（编入开局入参）→ modeling 主剧本开局。
