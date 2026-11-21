---
name: data-prior-survey
description: 数据领域建档问卷(mflowy)。用户上传数据后需要收集领域先验——目标列/标识列/单位/物理范围/受保护列——或提到"建档""数据问卷"时使用。清洗先验子剧本：data_profile(warm-up 证据) → 问卷 → eda(单变量+双变量+图) → dive-in 问卷(propose_questions 出题) → 收口产出 CleanPrior JSON 交棒 data-clean 开局（本剧本不开局，#233）。仅适用于已安装 mflowy CLI 的会话（`mflowy --help` 可用即在场）。
disable-model-invocation: true
---

# Data-Prior-Survey Skill：清洗先验问卷（warm-up → eda → dive-in → JSON 交棒）

本剧本是**先验子剧本**（零引擎：无转移分支/无奖励/无回放，也不开局），开局权在
主剧本 data-clean（#233）。你的职责：按剧本节奏引导问卷、把答案攒成
**CleanPrior JSON**、复述裁定后交棒 data-clean。**你不出清洗决策、不调开局
工具**；audit 分层——本剧本审计=对话本身，会话期审计=manifest 决策日志。

## 剧本四幕

### 1. data_profile（证据基质）

调 `data_profile(file_path)`——返回 df.info（形状/逐列 dtype/non-null 计数），
表与产物落盘归档。这是 warm-up 的出题依据。

### 2. warm-up 问卷（≤3-5 问，证据出题）

循环调 `propose_questions(file_path, card_partial)`：

- 每轮取 1-3 问按序问用户（必填项优先——必填集见 `mflowy propose_questions -h`；
  可选问由证据触发）；
- 已答字段并入 `card_partial` 回传（无状态——问卷状态在你手里）；
- 列名错当场被拒（fail-loud），把错误原样反馈用户修正；

### 3. eda + dive-in 问卷

调 `eda(file_path, target)`——单变量（typed 列统计 + 规则化 findings：id/常量/缺失/
单位疑似/物理范围锚）+ 双变量（三路关联 + 强/弱/未证档）+ 判读图族，表与图落盘归档。

再循环 `propose_questions(file_path, card_partial, eda_findings=<双变量 findings 行>)`——
强相关对/未证对作为 dive-in 追问的证据（"这两个量物理上是同一回事吗？"）。

### 4. 收口交棒（JSON 货物）

- `card_partial` 攒齐必填与证据触发的可选项；**字段面零硬编码**——CleanPrior 的
  必填/可选/类型契约一跳发现自 `mflowy data_load_or_start -h` 的 `--prior` 描述
  （propose_questions 出题同源；domain/goal 纯叙事走开局显式参数，不进先验）；
- **交付物 = 开局货物 JSON**（CleanPrior 字段 + domain/goal 叙事）：逐项复述关键
  裁定让用户过目后交棒 **data-clean skill**——主剧本拿 JSON 经 CLI 传参开局；
- **本剧本不开局**：`data_load_or_start` 调用面归 data-clean；开局后环境冻结，
  改先验 = 新 episode（重开）。先验持久化面 = 会话 manifest——**无盘上卡工件**，
  JSON 只经对话交棒。

## 约束

- **题型语法**：答案必须可编译进 CleanPrior 字段（或开局显式参数）——落不进类型
  契约的问题不要问；
- 用户答不上来的项**宁缺勿编**——空值走默认，校验闸会拦不一致；
- 无 target 的探索性数据：流程中断不支持（YAGNI）——如实告知用户；
- 出题节奏由你控场（生成器不设上限，你按 3-5 问分批）；每问出示证据引用。

## 参考

- 工具发现：记不清工具名或参数面时，先 `mflowy --help` 看分组清单、`mflowy <tool> --help` 看全参与用法（--prior 嵌套字段面同此揭示）；
- 姊妹词表：statistic.univariate / statistic.bivariate（findings 溯源）；
- 主剧本：data-clean（owns data_load_or_start 与回合循环）。
