# POLL + REVISE — 反馈观测与修订（判据见骨架）

## 有序调用链

1. `get_competition_submission(ref)` → **提交后立即执行**——ERROR 在页面上只是角标，不查不知道
2. 有分赛：`get_competition_leaderboard(competitionName)` → 分数/排名差分
3. `search_competition_submissions(competitionName)` → 提交历史与逐条错误
4. `list_competition_topics(competitionName)` → 主办方澄清/评级公布；深读 `get_forum_topic`
5. 需核对时：`download_competition_submission(submissionId)` 取回已提交文件

## 反馈通道矩阵

| 想知道什么 | 工具 | 有分赛 | 无分人工赛 |
| --- | --- | --- | --- |
| 是否过校验、错误原文 | get_competition_submission | ✓ | ✓（唯一硬反馈） |
| 提交历史逐条错误 | search_competition_submissions | ✓ | ✓ |
| 取回已提交文件核对 | download_competition_submission | ✓ | ✓ |
| 分数/排名 | get_competition_leaderboard | ✓ | 空（常态） |
| 澄清/评级公布 | list_competition_topics / get_forum_topic | ✓ | ✓（唯一可能的评审反馈入口） |

## 事件信号（定性；数值化形式见 MDP.md Φ——π 读信号，账本由 submissions.json 承载）

| 事件 | 外环响应 |
| --- | --- |
| 校验失败 | 回 PACKAGE 最小改包；error_description 回流 package.md 规则表 + lint 实现各一条 |
| 分数提升 | 有效路径——可继续迭代或满足终止条件收局 |
| 分数不升/劣化 | 回 SOLVE 重做（或 PACKAGE，若文件内容问题） |
| 无反馈 | 不回退；降频扫描讨论区（可建议用户挂 cron），满足终止条件即收局 |
| 澄清改题 | 回 SPEC 重读题 |

## 陷阱

- 无分赛排行榜空是常态不是异常；poll 幂等，可 cron 化（如每日一次）
- 终止不以"收到评审反馈"为必要条件——人工评审可能永不到

## 出边

校验通过 ∧ 通道已扫 → DONE（收局报告：交付物、提交记录、账号披露、后续修订建议）；按事件类别回退（映射见骨架）
