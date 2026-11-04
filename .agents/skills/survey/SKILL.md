---
name: survey
description: 数据领域建档问卷(mflowy)。用户上传数据后需要收集领域先验——目标列/标识列/单位/物理范围/受保护列——或提到"建档""数据问卷""开始清洗"时使用。嵌套研究流的外层剧本：data_profile(warm-up 证据) → 问卷 → eda(单变量+双变量+图) → dive-in 问卷(propose_questions 出题) → clean_session_start(card) 直进清洗 POMDP。仅适用于已配置 mflowy MCP server 的会话。
disable-model-invocation: true
---

# Survey Skill：嵌套研究流外层剧本（warm-up → eda → dive-in → clean_mdp）

外层是**剧本协议**（零引擎：无转移分支/无奖励/无回放），内层才是 POMDP。
你的职责：按剧本节奏引导问卷、把答案攒成 `DomainCard`、开局内层清洗会话后交棒
（clean skill）。**你不出清洗决策**；audit 分层——外层审计=对话本身，内层=manifest
决策日志。

## 剧本四幕

### 1. data_profile（证据基质）

调 `data_profile(file_path)`——返回 df.info（形状/逐列 dtype/non-null 计数），
表与产物落盘归档。这是 warm-up 的出题依据。

### 2. warm-up 问卷（≤3-5 问，证据出题）

循环调 `propose_questions(file_path, card_partial)`：

- 每轮取 1-3 问按序问用户（必填 domain/target 优先；可选问由证据触发）；
- 已答字段并入 `card_partial` 回传（无状态——问卷状态在你手里）；
- 列名错当场被拒（fail-loud），把错误原样反馈用户修正；

### 3. eda + dive-in 问卷

调 `eda(file_path, target)`——单变量（typed 列统计 + 规则化 findings：id/常量/缺失/
单位疑似/物理范围锚）+ 双变量（三路关联 + 强/弱/未证档）+ 判读图族，表与图落盘归档。

再循环 `propose_questions(file_path, card_partial, eda_findings=<双变量 findings 行>)`——
强相关对/未证对作为 dive-in 追问的证据（"这两个量物理上是同一回事吗？"）。

### 4. 收口交棒

- `card_partial` 攒齐必填（domain/target）与证据触发的可选项；
- 调 `clean_session_start(file_path, card=<card_partial>)`——start 即全量校验
  （fail-loud）并富化 never_drop，直进动作面；
- 之后交棒 **clean skill** 回合循环；本剧本结束。

## 约束

- **题型语法**：答案必须可编译进 DomainCard 字段——落不进类型契约的问题不要问；
- 用户答不上来的项**宁缺勿编**——空值走默认，校验闸会拦不一致；
- 无 target 的探索性数据：流程中断不支持（YAGNI）——如实告知用户；
- 问卷完成于 `clean_session_start` 之前——开局后环境冻结，改卡 = 新 episode（重开）。
- 出题节奏由你控场（生成器不设上限，你按 3-5 问分批）；每问出示证据引用。

## 参考

- 姊妹词表：statistic.univariate / statistic.bivariate（findings 溯源）；
- 内层引擎工具：clean_session_start / clean_propose / clean_apply（clean skill 手册）。
