---
name: bo-suggestion
description: BO 推荐操作主剧本(mflowy)——owns 开局（bo_load_or_init 唯一调用面）与全循环（推荐→台账→湿实验→回填→学习信号）。签名 = <BOCard 可空> <湿历史 parquet|study db 至少一> <model_run_id>。用户要求"推荐下一轮实验/继续 BO/回填实验结果"时触发；无卡时代调 bo-prior。
disable-model-invocation: true
---

# BO-Suggestion Skill：BO 推荐操作主剧本（四幕 + 纪律）

bo = agent 调 BO 工具的**工作流**（非会话）：状态载体 = study sqlite db，预测模型 =
调用入参（换模型仅重置学习信号）。你的职责：开局 → 推荐轮循环 → 回填与信号；
湿实验在系统外（人工/自动化实验室，跨天）。

## 开局幕（一次性）

1. **剧本面预检**：BOCard（`<task_dir>/BOCard.yaml`，可空——无卡先**代调 bo-prior**
   产卡再回来）、湿历史 parquet 与 study db **至少一个**（两缺即拦——机器面
   fail-loud 兜底）、`model_run_id`（预测模型 run）；
2. **必问**：上轮信号/漂移情况（新局先验讨论必看旧局失败信号——最小闭环继承形态）；
3. **一次性核对（不逐轮）**：模型是回归（分类模型拒绝）、卡搜索维 ⊆ 模型 x_names、
   单位显式（列名后缀承袭上游清洗词表——不显式让用户补命名）；
4. 调 `bo_load_or_init(`study_name, model_run_id, data=, db=, target_column=,
   direction=, search_space=…读卡直传, band=（match 时）)`：init（历史建新 study，
   ≥20 行机器门）/ continue（既有 db；换模型自动归档重置信号）/ rebuild（空间组
   变更——迁移式，**迁移报告必须对话叙述**：迁/弃/终局化计数不静默）。

## 推荐幕（一轮两相，可多轮并存）

1. `bo_ask(`study_db, n_candidates=k×批缺省)`——k = 剪枝损耗补偿系数，你自判
   （k×批缺省直传候选数，卡零字段）；在途避让保证不重推未回填点；
2. **填钉值**：候选 csv 非搜索维空列——按卡规则/历史众数/领域常识直填并**标注假定**
   （无源或冲突必问用户，不静默编）；
3. `predict(`data=候选文件, model=model_run_id)` 补点值（bo 工具零触碰模型）；
4. **读卡逐行判 {keep, prune}**：硬规则 → 预测范围对目标带（**相交判定**——无交集
   才出带，骑带缘归你裁量）→ 多样性；违例候选 `bo_prune(`study_db, trials=,
   reason=hard_rule_violation|out_of_band|user_rejected)`；批内多于实验室容量时
   `bo_diversity_check(`study_db, candidates=keep 清单, batch_size=)` 收口多样批；
5. **推送台账** `<study 目录>/round_00N.csv`（每轮新文件；列 = trial/round/特征列
   /measured_value/status/attribution/label/source/replicate_of——**零预测列族**，
   预测值是你的过程工件）+ **标准尾部**（每轮必附）：
   - 非验证值提醒：预测值来自预训练代理模型，**未经实验验证**；
   - 信号期持续提示（学习信号触发后每次推送附「预测模型需要学习…」式提示，持续到
     信号重置）；上批漂移声明（「上批触发漂移告警，本轮推荐在 X 假设下给出」）。

## 湿实验幕（等待态，零工具）

等待用户执行台账（跨天）；可随时**回推荐幕续出轮**（在途避让铺开多轮并存、round
每轮新文件、trial 全局号跨文件无歧义）。

## 回填幕

1. **两路对账**：台账文件更新（status/measured_value 非空 = 回填行；label 非空 =
   已处理标记）或带序号消息（你解析：序号→trial 对账、自由文本抽取、歧义**追问**
   ——单位可疑/缺列/重复序号/序号对不上，不静默修不静默弃）；
2. **判定标签**：predict 复算 ŷ → 按卡容差判 close/severe_mismatch/not_counted
   （无容差主观判断 + **用户核实后才回填**）；**负向归因用户确认**（合理才计数）；
3. **deviated 变体确认**（实测参数 ≠ 推荐参数）：问用户是推荐变体（原 trial 关
   FAIL、新行来源=推荐）还是自发追加（原 trial 留在途、新行来源=自发追加）；
4. **批量回填**：一批台账多行 = 一次 `bo_tell(`study_db, rows=…, band=（match 读卡）,
   positive_gate=, negative_gate=, window_k/m=（读卡传门限）)`——平行样首条 done +
   余条 appended(replicate_of)；放弃/操作事故按行形态申报；
5. **回写台账** label/source/replicate_of（已处理标记）；**信号批末报告不打断**——
   触发后：尾部提示开始持续 + 可选诊断叙述一次（素材 = 信号统计 + 台账对照 +
   剪枝审计）+ 建议（继续推荐 / 交棒更新模型 = 人点火——上游链重跑后
   `bo_export_dataset(`study_db, output_path=)` 按需协助导出累计湿数据，模型就绪
   后带新 run_id 重进本剧本）。

## 纪律

- **卡只读**：卡改动（身份字段或策略组）一律经 bo-prior（改卡发生地与提醒单源）；
- 已终态 trial 再操作会被工具拒绝（两路回填先到为准）——对账冲突时向用户报告已
  回填内容，无覆盖入口；
- 台账序号 = trial number 直用，跳号即事实（PRUNED/FAIL 也占号）；
- 纯 Virtual 连续筛选 = 本主循环的用法（回填缺席、多轮 ask 铺开）——无模式开关，
  虚拟预测值永不入 study。

## 参考

- 六工具参数面与行形态细则：`tools/list`（docstring 即 API 文档）；
- 系统闭环设计与落位对照：[bo 域 README](../../../packages/builtin_plugins/bo/README.md)；
- 词面（BOCard/study/trial/推荐台账/湿实验/学习信号）：CONTEXT.md BO 节；
- 先验子剧本：bo-prior（产卡/改卡唯一发生地）。
