---
name: kaggle-competition
description: Kaggle 竞赛外环控制回路(看题→数据→做题→打包→preflight→提交→反馈→修订)。用户要求参加/完成/打某个 Kaggle 竞赛,或单独要求下载竞赛数据、提交竞赛包、查询提交反馈与错误时使用。适用于任意题型;求解方式用户指定则从其指定,否则由 agent 按题自行判断。仅适用于已配置 kaggle MCP 的会话。
---

# Kaggle Competition：外环控制回路（骨架）

竞赛解题是外环 MDP：提交是稀缺动作（预算每日 5 次），平台反馈是一等观测，preflight 把平台约束本地化——不把预算花在可本地判定的失败上。本文件是**常驻骨架**（状态机+判据+回退映射+不变量）；各阶段的动作与有序工具链住 `references/<阶段>.md`，**进入一个阶段 ≡ 已读该阶段的 reference**（各阶段判据第一条）。形式化（S/A/P/R/T）见 [MDP.md](MDP.md)；lint 活脚本在 `scripts/submission_lint.py`。

## 状态机（正向边 = 判据触发，π 判定；判据齐备才出边）

| 阶段 | 判据 |
| --- | --- |
| SPEC 看题 | 已读 references/spec.md ∧ 一句话说清交付物/评审方式/deadline ∧ 文件清单已记录 ∧ 榜单与讨论区已扫 |
| DATA 拉资产 | 已读 references/data.md ∧ 数据落盘且逐文件字节数与 listing 一致（stand-in 分支：PROVENANCE 如实，规则门 pending 不阻塞后续阶段） |
| SOLVE 做题 | 已读 references/solve.md ∧ 本地验证门全绿 ∧ 全部交付物一键再生 |
| PACKAGE 打包 | 已读 references/package.md ∧ lint 0 ERROR |
| SUBMIT 提交 | 已读 references/submit.md ∧ 预算有余 ∧ 规则门已过 ∧ 包指纹 ≠ 曾失败指纹 → 拿到提交 ref（技术失败无 ref 不扣预算；校验 ERROR 已扣） |
| POLL 反馈 | 已读 references/poll-revise.md ∧ 反馈通道查过一轮 |

## 回退映射（回退边 = 反馈类别触发，环境判定）

| 反馈类别 | 回退点 |
| --- | --- |
| 校验失败（ERROR + error_description） | PACKAGE 最小改包；新 error_description 回流一条 lint 规则 |
| 有分赛评分异常 | SOLVE 重做（或 PACKAGE，若文件内容问题） |
| 无反馈（人工评审/排队，榜空是常态） | 不回退——降频扫描讨论区，满足终止条件即收局 |
| 讨论区主办方澄清/改题 | SPEC 重读题 |

## 不变量（跨阶段，全程生效）

- **预算纪律**：任何 submit 前必有零成本 preflight；失败后不烧新预算重试同包
- **提交后立即 poll**：校验结果不在提交响应里，ERROR 在页面只是角标
- **终止**：校验通过 ∧ 反馈通道已扫——不以"收到评审反馈"为必要条件（人工定性赛可能永不到）；deadline 是硬壁，日预算耗尽且无 revise 空间 → 中止次日续
- **方法中立**：求解方式用户指定则从其指定，未指定按题型自选——回路不绑定任何技术栈
- **escalate**：规则门（平台人工点击，不可无头绕过）与评审申诉是唯一通往人的出口
- **诚实性**：PROVENANCE 如实（stand-in 与官方数据不得混淆）；题目声称但数据不存在的缺陷如实记 verified-clean 不制造修复；账号上创建的 kernel/提交须在总结列明
