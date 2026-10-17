---
name: model-brief
description: 建模协议问卷(mflowy)。清洗收局后、开始建模前使用——用户要求"建模/训练模型"且清洗已完成时触发。规则预填候选，用户只确认四项必答，剧本层产出 ProtocolCard（对局规则卡）后 modeling_session_start(protocol_card) 直进建模 MDP。前置:clean skill 已 finish。
---

# Model-Brief Skill：建模协议问卷（剧本层谈判）

三段式研究流第三段的前置：`建档 → 清洗 MDP →(冻结)→ **model-brief 问卷** → 建模 MDP`。
你的职责：把"对局规则"问清楚、编译成 `ProtocolCard`——**你不做结构决策**，那是建模
MDP 回合里的事（model skill）。

## 一段式：规则预填 + 用户确认

与 survey 的两段式不同——这里的大多数字段可从数据事实与清洗产物**规则推导**，你预填、
用户确认或改。**必问四项**（规则给不出答案的）：

1. **建模成功判据**（goal）：业务口径一句话（"MAE≤X""比现有流程好 10%"）——
   区别于 survey 已收的研究目标；
2. **指标偏好**：按任务预填（回归 MAE / 分类 F1），用户可改（如筛查场景改 recall）；
3. **预算**：max_steps（缺省 10——每动作吃一步：run 或 FE 换档，π 在回合里自行分配深度与 FE 探索）；
4. **外部基线**（external_baseline）：有无既有基准分数要对标（没有则空）。

候选面（candidate_families）来自 **research-review skill**（文献证据：相似问题→模型
候选）——本问卷不重复出题，用户直接确认 research-review 产物即可；无 research 产物时
留空（全部已声明族开放）。

### 预填项（证据来自 clean_finish 产物，不重跑画像）

- `task`：由 target dtype 与统计推断（数值/唯一率/步长判据）——唯一归属本协议卡；
- `cv`：n<1k → 3~5 折；分类任务受最小类计数约束（cv ≤ min class count）。

## 收口：编译交棒

攒齐必答后在剧本层组装 ProtocolCard 并调
`modeling_session_start(final_path, card, protocol_card)`——start 即四类相容校验
（task↔target dtype / metric↔task / cv↔数据事实 / 预算与先验机械校验：候选 ⊆ 已声明
族目录、partial ∈ 资源边界）fail-loud + D 物化冻结 + 预算起算，直进 run 动作面。
校验失败把错误**原样**反馈用户修正；开局后协议不可改，改协议 = 新 episode。
数据源 = 清洗收局的 `final_path`（交接单点：`final_path` → 开局数据源；`plan_yaml`
是审计/重放工件，开局不消费——交接路径见 research-flow 第二幕交付物 → 第三幕开局）。

## 纪律

- 宁缺勿编：用户答不上来的约束留空（默认全族开放）；
- 问卷是建模 MDP 的**参数化**（对局规则），不是特征工程决策（那是回合里的事）；
- 词表：协议卡字段即 modeling_session_start 的冻结分量，不发明字段。

## 参考

- 类型与校验闸：`packages/model/mflowy/model/protocol.py`（ProtocolCard.build）
- 候选收窄：research-review skill（文献先验——banned_families 语义已被
  candidate_families 取代，禁用改为"不进候选清单"）
