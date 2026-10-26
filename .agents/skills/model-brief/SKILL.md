---
name: model-brief
description: 建模协议问卷(mflowy)。清洗收局后、开始建模前使用——用户要求"建模/训练模型"且清洗已完成时触发。规则预填候选，用户只确认四项必答，剧本层产出开局入参（cv_card = cv.split 配置 yaml 明文 + 显式协议参数）后 modeling_session_start 直进建模 POMDP。前置:clean skill 已 finish。
---

# Model-Brief Skill：建模协议问卷（剧本层谈判）

三段式研究流第三段的前置：`建档 → 清洗 POMDP →(冻结)→ **model-brief 问卷** → 建模 POMDP`。
你的职责：把"对局规则"问清楚、编译成开局入参（**cv_card** = cv.split 配置 yaml 明文，如 `cv: 3\ntest_size: 0.2`——划分构建参数之外的协议走 start 显式参数/card 背景）——**你不做结构决策**，那是建模
POMDP 回合里的事（model skill）。

## 一段式：规则预填 + 用户确认

与 survey 的两段式不同——这里的大多数字段可从数据事实与清洗产物**规则推导**，你预填、
用户确认或改。**必问四项**（规则给不出答案的）：

1. **建模成功判据**（goal）：业务口径一句话（"MAE≤X""比现有流程好 10%"）——
   区别于 survey 已收的研究目标；
2. **指标偏好**：按任务预填（回归 MAE / 分类 F1），用户可改（如筛查场景改 recall）；
3. **预算**：max_steps（缺省 10——只有计算动作吃步：run_hpo / final_train_test；
   配置编辑零步费，π 在回合里自由参数化）；
4. **外部基线**（external_baseline）：有无既有基准分数要对标（没有则空）。

候选面（candidates）来自 **research-review skill**（文献证据：相似问题→结构化模型
候选 `[{family, baseline, space_hints, seed}]`）——本问卷不重复出题，用户直接确认
research-review 产物即可；无 research 产物时留空（全部已声明族开放，首个为 baseline）。
校验闸会在 start 时核对：baseline 恰一、hints ⊆ 族签名声明空间。

### 预填项（证据来自 clean_finish 产物，不重跑画像）

- `task`：由 target dtype 与统计推断（数值/唯一率/步长判据）——唯一归属本协议卡；
- `cv`：n<1k → 3~5 折；分类任务受最小类计数约束（cv ≤ min class count）。

## 收口：编译交棒

攒齐必答后在剧本层组装开局入参并调
`modeling_session_start(final_path, cv_card=<split yaml 明文>, target=, task=,
candidates=, max_steps=, id_columns=, card=<DomainCard 背景>)`——显式参数 > card
分层缺省，合并结果冻结进 manifest；start 即四类相容校验
（task↔target dtype / metric↔task / cv↔数据事实 / 预算与候选机械校验：候选 ⊆ 已声明
族目录、baseline 恰一、hints ⊆ 签名空间）fail-loud + D 物化冻结 + 预算起算，直进
八动作面。校验失败把错误**原样**反馈用户修正；开局后协议不可改，改协议 = 新 episode。
数据源 = 清洗收局的 `final_path`（交接单点：`final_path` → 开局数据源；`plan_yaml`
是审计/重放工件，开局不消费——交接路径见 research-flow 第二幕交付物 → 第三幕开局）。

## 纪律

- 宁缺勿编：用户答不上来的约束留空（默认全族开放）；
- 问卷是建模 POMDP 的**参数化**（对局规则），不是特征工程决策（那是回合里的事）；
- 词表：开局冻结分量 = cv_card 键（⊆ cv.split 参数面）+ start 显式参数，不发明字段；多目标 = 每目标各开一个单目标任务会话（三层循环，不进本协议）。

## 参考

- 类型与校验闸：`packages/builtin_plugins/model/mflowy/model/protocol.py`（CVCard.build）
- 结构化候选：research-review skill（文献先验——candidates 带 baseline 锚点 /
  space_hints 注入搜索起点 / 候选 seed）
