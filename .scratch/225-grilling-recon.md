# #225 Round 2 grilling 现状勘察底稿（七问备料，2026-11-17）

> 执行票 #231 会话内并行备料——grilling 需用户参与裁定，本文件只载现状事实与
> 备选口径，不作裁定。基线 = dev 2fe431b1（#226 三切片已收口）。

## 现状总览（#226 落地后的新地基）

- **Observation 四件**（#234）：o.prior / o.biz / o.actions / o.validation——
  收账/归属类问题的观测挂点已就位（o.biz 域自定形状）
- **两域会话实体化**（#235/#236）：CleaningSession / ModelingSession（typed 六操作）；
  sid = 会话目录绝对路径；决策日志补 r 已落
- **判读退役**（#236）：机器 findings → 统计证据（`modeling_statistics` 独立 compute
  工具，run 寻址，走 JobProvider）；findings 单源 source=llm
- **工件契约**（#236）：oof_y_pred.parquet 带 indices 列 + 源数据指纹 tag

## Q1 收账机制（hpo 后 agent 怎么取数）

**现状**：`run_hpo` 响应 = Observation（board 摘要进 o.biz）；`observe_modeling`
读时投影 findings/账本摘要/face。统计证据 = 独立工具 `modeling_statistics(sid, key)`
（run 寻址重构 X/y/y_pred，产证据表 + 图集落统计 run）。
**#226 已裁预定**（挂起标注节）：回读 = 查询式集成进动作——hpo 动作内两分支归一
（remote = 拉远程 zip → 解包 → 查询 run / local = 查询 run），复用
`download_platform_job_artifacts` + `get_run(tracking_uri)`；无 sidecar 新契约、
无 JobOutcome 第三形态。
**待裁面**：收账是否需要会话外查询面（跨进程/跨会话取 trial 明细），还是恒在
动作响应 + observe 内消化。

## Q2 study 续跑 resume 统一

**现状**：model 域 `_open_study`（session.py）= 增量续跑（目标同一性 = objective_id：
空间变增量继续、seed/scoring 变重开）；remote 面有独立 `resume_study_from`
（job_provider 协议 + local 实现 + modeling 工具入参）。
**张力**：两套 resume 语义（会话内 study 增量 vs remote plan 级 resume_study_from
参数）——统一口径待裁（同一性判据是否共享 objective_id 面）。

## Q3 平台实验归属

**现状**：会话锚定实验 = `modeling_session_{sid}` / `clean_session_{sid}`
（`_session_experiment` get-or-create，工件根 = 会话目录 .mlruns）；DAG 节点 run
经 `set_active_experiment` 挂进；remote provider 场景实验归属平台侧
（`ensure_platform_experiment`）。
**待裁面**：remote 计算的 run 归属——回本地会话锚定实验（收账拉回后 re-log）还是
平台实验为真源、会话只持 run_id 引用。

## Q4 步费与失败语义

**现状**（#234 落地）：计步 = 壳按 counts_step 在 apply commit 缝单源记账；
body 内失败即免费（零日志零观测变更，孤儿工件无害）；计步后 body 失败 =
内存回退不落盘（免费）。**无「失败吃步」路径**——失败恒免费。
**待裁面**：remote submitted 异步段的步费时点（提交即吃 / finished 收账才吃 /
失败退款）与 submitted→failed 的语义。

## Q5 在途生命周期门

**现状**：optuna trial RUNNING 态在 bo 域有完整生命周期（ask 建 RUNNING →
prune/tell/diversity 终局化；迁移式重建时在途终局化 FAIL）；model 域 DAG run
无在途门（JobOutcome submitted 形态在 #156/#197 已建但会话终训恒 local——
`_final_train_test` 显式只受理 finished）。
**待裁面**：会话内 submitted run 的生命周期门（是否允许并行多个在途、在途时
apply/finish 的门、崩溃恢复在途的终局化策略）。

## Q6（R1 已裁面复述）全计算工具走 JobProvider

已裁：remote 不阻塞（submitted/finished 分支住 SKILL，agent 轮询）；`modeling_statistics`
已按 R1 走 provider（#236 落地）。余下：hpo 动作本身 remote 化的受理面。

## Q7 submitted 信封 watch 命令

**现状**：#234 落地二联信封 {sid, observation}；watch 命令未实现
（modeling tools TODO 注释占位——「#225 remote 半边随用随加」）。
**待裁面**：watch 命令形态（独立工具 vs 信封字段自带 shell 命令文本）与轮询语义
（长轮询/即时返回状态）。

## 勘察锚点

- `packages/builtin_plugins/model/mflowy/model/session.py`：_open_study/objective_id/
  _session_experiment/_final_train_test（恒 local 受理注释）
- `packages/builtin_plugins/model/mflowy/model/job_provider/`：resume_study_from 协议面
- `packages/builtin_plugins/bo/mflowy/bo/tools.py`：在途 trial 生命周期先例
- `packages/driver/mflowy/driver/session.py`：apply_ctx 计步缝（Q4 现状）
