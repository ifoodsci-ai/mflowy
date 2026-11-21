---
name: model-prior-survey
description: 建模协议问卷(mflowy)。清洗收局后、开始建模前使用——用户需要确认建模对局规则（成功判据/指标/预算/外部基线）或提到"建模问卷/建模协议"时触发（"建模/训练模型"触发词归 modeling 主剧本）。规则预填候选，用户只确认四项必答，剧本层产出开局入参 JSON（cv_card = cv.split 配置 yaml 明文 + 显式协议参数）交棒 modeling 开局——本剧本不开局，#233。前置:data-clean skill 已 finish。
disable-model-invocation: true
---

# Model-Prior-Survey Skill：建模协议问卷（剧本层谈判，JSON 交棒不开局）

三段式研究流第三段的先验子剧本：`建档与清洗 POMDP →(冻结)→ **model-prior-survey
问卷** →(JSON 交棒)→ 建模 POMDP（modeling 主剧本开局）`。你的职责：把"对局规则"
问清楚、编译成**开局入参 JSON**（cv_card = cv.split 配置 yaml 明文，如
`cv: 3\ntest_size: 0.2`——划分构建参数之外的协议走 start 显式参数）——**你不做
结构决策、不调开局工具**（modeling_load_or_start 调用面归 modeling，#233），
结构决策是建模 POMDP 回合里的事。

## 一段式：规则预填 + 用户确认

与 data-prior-survey 的两段式不同——这里的大多数字段可从数据事实与清洗产物
**规则推导**，你预填、用户确认或改。**必问四项**（规则给不出答案的）：

1. **建模成功判据**（goal）：业务口径一句话（"MAE≤X""比现有流程好 10%"）——
   区别于建档问卷已收的研究目标；
2. **指标偏好**：按任务预填（回归 MAE / 分类 F1），用户可改（如筛查场景改 recall）；
3. **预算**：max_steps（缺省 10——只有计算动作吃步：run_hpo / final_train_test；
   配置编辑零步费，π 在回合里自由参数化）；
4. **外部基线**（external_baseline，建模域先验）：有无既有基准分数要对标——有则
   问四件套 `{value: 分数, metric: 口径（须与本任务指标同域）, source_text: 出处
   （文献引用/内部实验记录）, source: literature/internal_history/manual}`（权威度
   白名单外 start 会 fail-loud），没有则空。

候选面（candidates）来自 **model-research-review skill**（文献证据：相似问题→
结构化模型候选）——本问卷不重复出题，用户直接确认 model-research-review 产物
即可；无 research 产物时留空（全部已声明族开放，首个为 baseline）。候选字段面
**零硬编码**——一跳发现自 `mflowy modeling_load_or_start -h` 的 `--candidates`
描述（start 校验闸同源：baseline 恰一、hints ⊆ 族签名声明空间）。

### 预填项（证据来自 data_finish 产物，不重跑画像）

- `task`：由 target dtype 与统计推断（数值/唯一率/步长判据）——唯一归属本协议卡；
- `cv`：n<1k → 3~5 折；分类任务受最小类计数约束（cv ≤ min class count）。

## 收口：编译交棒（JSON 货物）

攒齐必答后在剧本层组装**开局入参 JSON**（cv_card 明文 + target/task/metric/
candidates/max_steps/id_columns/goal/external_baseline），逐项复述关键裁定让用户
过目后交棒 **modeling skill**——主剧本拿 JSON 经 CLI 传参开局（file_path =
清洗终态路径；显式参数 > card 背景，合并结果冻结进 manifest）。开局货物字段面
**零硬编码**——一跳发现自
`mflowy modeling_load_or_start -h`。start 即四类相容校验
（task↔target dtype / metric↔task / cv↔数据事实 / 预算与候选机械校验）fail-loud
+ D 物化冻结 + 预算起算。校验失败把错误**原样**反馈用户修正（改协议回到本问卷）；
开局后协议不可改，改协议 = 新 episode。数据源 = 清洗收局的 `final_path`
（交接单点：`final_path` → 开局数据源；`plan_yaml` 是审计/重放工件，开局不消费
——交接路径见 research-flow 第二幕交付物 → 第三幕开局）。协议的持久化面 =
会话 manifest——**无盘上卡工件**，JSON 只经对话交棒。

## 纪律

- 宁缺勿编：用户答不上来的约束留空（默认全族开放）；
- 问卷是建模 POMDP 的**参数化**（对局规则），不是特征工程决策（那是回合里的事）；
- 词表：开局冻结分量 = cv_card 键（⊆ cv.split 参数面）+ start 显式参数，不发明字段；多目标 = 每目标各开一个单目标任务会话（三层循环，不进本协议）。

## 参考

- 工具发现：记不清工具名或参数面时，先 `mflowy --help` 看分组清单、`mflowy <tool> --help` 看全参与用法（--cv-card/--candidates 嵌套字段面同此揭示）；
- 结构化候选：model-research-review skill（文献先验——candidates 交棒货物）；
- 主剧本：modeling（owns modeling_load_or_start 与回合循环）。
