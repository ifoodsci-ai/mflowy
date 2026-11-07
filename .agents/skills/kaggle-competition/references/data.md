# DATA — 拉资产（判据见骨架）

## 动作与有序工具链

**数据集 `download`**：
1. `get_competition_data_files_summary(competitionName)` → 文件面概览（可选）
2. `download_competition_data_files(competitionName)` → 全量落盘
3. 本地逐文件字节数核对 vs SPEC 记录的清单

**规则门（403 "must accept rules"）**：无法无头绕过（内部 API 需浏览器会话 + reCAPTCHA，勿再试）——escalate：把规则页 URL 交给用户点击，异步等待。**等待期 stand-in 分支**：按官方文档 schema 造同构数据先把全管线跑通，用户点击后真实数据零改动重跑。造法（方法论，per-赛手工活，不脚本化）：

- 读 SPEC 记录的文件 schema 与字段语义，逐列造：数值列覆盖量级/缺失/物理范围，类别列保留基数与分布形状，时间列保留跨度与频率，id 列保持唯一
- 行数与官方 listing 一致（校验器按行数判）；PROVENANCE **必须**如实标注 stand-in

**代码框架 `fetch_baselines`**：
1. `search_notebooks(competition=X, sortBy=VoteCount)` → 公开 baseline/starter 候选
2. `get_notebook_info(userName, kernelSlug)` → 元数据
3. `download_notebook_output(ownerSlug, kernelSlug)` → 参考实现

**情报增量**：`list_competition_topics` / `get_forum_topic` 补 SPEC 第一遍的漏。

## 陷阱

- ⚠️ MCP `save_notebook` 挂竞赛数据源会**静默丢弃**——kernel 通道一律走 v1 API push（`/api/v1/kernels/push`，Bearer 可用；输出经 `/api/v1/kernels/output` 取回）
- 规则门只阻塞两条边（真实数据 download 与最终 submit）——stand-in 下 SOLVE→PACKAGE 照常推进，不是全回路冻结

## 出边

字节一致 → SOLVE（读 references/solve.md）；403 → escalate（异步，不阻塞 stand-in 路径）
