# SPEC — 看题（判据见骨架，此处不复述；本文件 = 该阶段的动作面与陷阱）

## 动作与有序工具链

**摸清赛题 `query`**：
1. `search_competitions(search=题名)` → 定位 competitionName
2. `get_competition(competitionName)` → 元数据：deadline / reward / category
3. `list_competition_pages(competitionName)` → 逐页读规则与评价方式——**评审方式（自动分 or 人工定性）决定奖励模型与终止判读，deliverables 清单决定包结构**，deadline 定硬壁

**情报面第一遍扫**：
4. `list_competition_data_files(competitionName)` → 数据文件清单**含字节数**（DATA 阶段逐文件校验的基准）
5. `get_competition_leaderboard(competitionName)` → 有分赛看水位
6. `list_competition_topics(competitionName, sortBy=Hot 或 Recent)` → 主办方早期澄清可能改写题意；深读 `get_forum_topic(forumTopicId, includeComments=true)`

## 陷阱

- 评审方式没确认就往下走是最大的浪费——它决定 SOLVE 的方法选择（有分赛围绕指标做，人工评审赛围绕可复现工作流做）与 PACKAGE 的必备件清单
- 三件事（评审方式/deliverables/deadline+文件清单）要**落盘**到工作目录 SPEC 笔记——后续判据依赖的是记录而非上下文记忆

## 出边

判据齐备 → DATA（读 references/data.md）
