---
name: research-review
description: 文献候选评审(mflowy)。model-brief 问卷前使用——用户确认开始建模且需要文献依据选模型候选时触发。相似/跨领域研究问题 → 模型候选清单（candidate_families），经 ProtocolCard 进建模 MDP。前置:clean skill 已 finish。
---

# Research-Review Skill：文献候选评审（剧本层，接 model-brief 之前）

职能：把"选哪些模型族"的第一轮淘汰交给**文献先验**而非暴力筛选。产物只有
`candidate_families` 一个字段，交给 model-brief 编译进 ProtocolCard——本 skill
不碰数据、不碰引擎、不产出模型。

## 相似研究问题 → 模型候选

1. 从 DomainCard（domain/goal/target 语义）+ 清洗收局报告提炼**研究问题画像**：
   数据规模 / 特征形态（表格·宽窄·类别占比）/ 任务类型 / 量级约束；
2. 检索同领域与跨领域的相似问题基准与方法（Kaggle 同型赛题、领域 survey、
   仓库 examples/）；
3. 产出**候选清单**（对齐运行期族目录——`modeling_propose` 的 candidates 即
   已声明 `.mdp_family` 者；清单外族 = 第一轮淘汰）：
   - 每候选一句文献依据（"该规模表格基准上 GBDT 系长期占榜"）；
   - 无证据支撑的族不入清单（宁缺勿编——留空则全部已声明族开放）。

## 收口：交给 model-brief

产物随必答项一起进 ProtocolCard：`{candidate_families: [候选清单]}`——开局
校验（候选 ⊆ 已声明目录）fail-loud，坏值原样反馈修正。

## 纪律

- 本 skill 的全部产出是一个字段——不做特征工程、不做超参决策（那是回合里的事）；
- 文献依据可追溯：每候选一条出处或机理陈述，用户可否决；
- 候选清单是**开局边界**：清单外族被引擎拒绝（守卫拦"事实不成立"）——想跑某族
  就应在清单里声明它；
- 资源深浅（树数/epoch）不经本 skill——资源 = 签名参数，π 在回合里按空间摘要
  自行 pin（#33 资源机制全退）。

## 参考

- 消费方：model-brief skill（编译进 ProtocolCard）→ modeling_session_start
- 族目录声明：`packages/model/mflowy/model/registry.py`（`.mdp_family` = 目录标记 + structure）
