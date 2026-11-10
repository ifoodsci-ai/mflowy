# E2E 验收 runbook——RemoteJobProvider 接入上游实验服务

> [E2E 验收 runbook 票](https://github.com/ifoodsci-ai/mflowy/issues/165) 产物（task 票，2026-11-11）。
> **地位**：remote 实现系列 PR 的合并门禁——每 PR 标注必过阶段（见文末汇总表），所涉阶段全绿才可合；系列收尾跑全链。决策地图 [Wayfinder: RemoteJobProvider 接入上游实验服务](https://github.com/ifoodsci-ai/mflowy/issues/153) 不出实现代码——本文是**验收判据文本**，可执行驱动脚本随首个实现 PR 落仓（对齐既有 `make e2e` / `scripts/_e2e_*.py` 模式）。
>
> **阶段徽标**（上游依赖维度；素材维度见下）：
> - 🟢 = 实现落地后即可执行，上游无阻塞
> - 🔴 = 实现落地后仍被 [mflowy-platform#46](https://github.com/lincode7/mflowy-platform/issues/46) 阻塞（CLI scope 表缺 jobs/presets 路），上游扩表后转正
>
> 素材维度：S0/S1 以基座（[镜像与 preset 端到端验证基座](https://github.com/ifoodsci-ai/mflowy/issues/161)）即可执行；S2+ 需对应实现 PR 落地。#46 转正前，部分门 = S0/S1/S2/S5/S7 + S3/S4/S6/S8 的无上游依赖断言。

## 契约单源

本 runbook 只断言**可观察行为**，契约细节以各决议票为准：协议与参数面（[#156](https://github.com/ifoodsci-ai/mflowy/issues/156)）、client 与错误族（[#157](https://github.com/ifoodsci-ai/mflowy/issues/157)）、包装器与 zip 往返（[#158](https://github.com/ifoodsci-ai/mflowy/issues/158)）、工具组与观察面（[#159](https://github.com/ifoodsci-ai/mflowy/issues/159)，含 [单参推导回填](https://github.com/ifoodsci-ai/mflowy/issues/159#issuecomment-6771610683)）、产物清单（[#155](https://github.com/ifoodsci-ai/mflowy/issues/155)）、词汇（[#160](https://github.com/ifoodsci-ai/mflowy/issues/160)）、基座事实（[#161](https://github.com/ifoodsci-ai/mflowy/issues/161)）。
注：runbook 票题面写「本地两参查询（tracking_uri + artifact_root_local）」——已被 #160 收敛为**单参 tracking_uri 推导式**，本文按终稿执行。

## S0 前置与基座（每轮执行，🟢 现在即可跑）

**环境**：

- gateway `http://localhost:30080`（`MFLOWY_BASE_URL`，本地 NodePort）
- `MFLOWY_PLATFORM_JOB_PRESET=cpu-standard`（种子 **v2**，id `c0000004-0000-0000-0000-000000000004`：training-job:latest 8c/8Gi / workspace 1024Mi / 3600s）
- CLI 凭据 `~/.mflowy/credentials.json`（`cli-acceptance@mflowy.dev`）；OTP 验收账号 `mflowy-base@mflowy.dev`
- `MFLOWY_TELEMETRY=off`（自动化绕 elicitation 同意门；正式适用面 = 遥测票，未决）

**每轮步骤**：

1. 镜像重建：待测分支（含 `feat/remote-job-image` 的 `docker/Dockerfile`——zip/unzip + 烧 `MFLOWY_TASK_DIR=/user-data/workspace`、`MLFLOW_TRACKING_URI=sqlite:////user-data/workspace/mlflow.db`、`HOME=/tmp`）构建 `training-job:latest` → `docker save` → k3s containerd 导入；tar 留档 `/tmp/training-job-<git短sha>.tar`（#161 模式）。
2. 镜像内断言：四烧 ENV 回显、mflowy 版本回显（`importlib.metadata`，#161 实证形态）与 `mflowy list_nodes` 词表可用；`mflowy-job`（wrap 子命令）在 PATH——**仅切片 ④ 落地后**断言（基座镜像无此入口）。
3. preset 断言：cpu-standard v2 在列且 workspace 1024Mi（OTP web token 查 `GET /api/v1/job-presets`——该路对 web token 的可达性 #161 **未留档**，#161 驱动是以种子 id 直提 CreateJob；不通则以 S1 CreateJob 受理成功为替代证据并记录）。
4. OTP 登录演练：`POST /api/v1/auth/send-verification-code/login` → dev Redis（容器 `mflowy-infra-redis-1`，密码取 `mflowy-platform/.env.redis`）`vcode:login:<email>` 取码 → `POST /api/v1/auth/login`。**连续登录间隔 ≥65s**（per-IP 1/min + per-(purpose,email) 冷却，#161 踩坑两次）。

**驱动方式**：平台面 = curl 脚本（改编 #161 留档 `/tmp/mflowy-base-verify/run.sh`）；工具面 = MCP 通道小驱动脚本（stdio 连 `mflowy-mcp`，cwd 置于临时目录——通道纪律断言依赖真实 MCP 通道）。载荷构造：CLI runner 单 JSON argv、name/desc 等客户端拼进 JSON（无 key=value 追加合并）。

每 PR 常规门（不在本 runbook 范围）：`make test` / `make typecheck` / `make lint`。

## S1 平台契约回归（🟢 现在即可跑，OTP web token）

复跑 #161 基座验收（cmd=sh 直跑，不经 wrapper）——钉住**上游平台契约**，与 mflowy 实现解耦，捕获上游漂移（先例：#45 网关 regen 缺失、#46 scope 表）：

1. OTP 建实验 → presigned 上传 3 文件（task.sh / diabetes.py / payload.json，见附录）→ CreateJob（cmd=sh，cpu-standard v2）→ 轮询至终态 `succeeded`。
2. 断言：
   - ENV 契约回显：`MFLOWY_TASK_DIR` / `MLFLOW_TRACKING_URI` / `MFLOWY_JOB_ID` / `HOME=/tmp`
   - `mflowy modeling`（diabetes 最小链）rc=0
   - zip 往返：outputs 下 `{job_id}.zip` 可见可下载；含 `mlflow.db` + `.mlruns/` 树 + 模型工件
   - workspace 语义：`.scratch` 下载 404、ListFiles 无 workspace 根（emptyDir 不落 S3）
   - uploads 只读探针（写探针失败）

## S2 实验就绪：`ensure_platform_experiment`（🟢 实现后即测；upload-url 属 /experiments 下 files 面——#161 留档为「`/experiments/*` ✓」整体结论，files 子路未逐路实测，复测 403 则随 #46 转 🔴）

**正向**：

- 不传 id、`name` 给定 → 创建；返回含 `platform_experiment_id`、`files.uploaded[]`（`{local, remote: /user-data/uploads/<basename>}` 映射）、`inheritable: []`（新实验）
- 重复调用同清单 → 全部 `skipped`（MD5↔etag 幂等）；改动一个文件内容再调 → 恰该文件 `updated`
- 传 id（刚建的）→ 查询模式返回、零副作用
- J1 产出后再调 → `inheritable` 出现该 job（newest first、带 status 不过滤）

**负向（fail-loud 家族）**：

- 传不存在的 id → 404 报错带 id（**绝不静默转创建**）
- 同传 id 与 name → 参数来源冲突拒绝
- 不传 id 也不传 name → fail-loud
- 清单含目录 / basename 以 `.` 开头（dotfile）/ 清单内撞名 → fail-loud（dotfile 拒整次同步）
- MCP 通道传相对路径 files → 拒收 fail-loud（runner 通道接受——通道纪律）

## S3 提交与受理（🔴 blockedBy #46：create_job + list_presets）

**正向**：

- `switch_job_provider(None)` → 返回列表含 local（active）与 remote（available，依据 `MFLOWY_BASE_URL` 已配）；`switch_job_provider("remote")` → active=remote（进程级）
- hyperparameters_search（diabetes 载荷，见附录）remote 模式：`platform_experiment_id=E`、无 inherit → **立即**返回（remote 永不阻塞等待）JobOutcome：`status=submitted` + `job_id=J1` + `platform_experiment_id=E`、`dag_result=None`
- job 面形状（web token 或 get_job）：`cmd="mflowy-job"`、`args=["wrap","--tool","hyperparameters_search","--kwargs-json",<单 argv JSON>]`（#158 提交面；`--inherit-from` 仅 inherit 非空时出现）

**负向**：

- remote 缺 `platform_experiment_id` → fail-loud 提示「先调 ensure_platform_experiment」
- `MFLOWY_PLATFORM_JOB_PRESET` 未设 / 名不命中 → fail-loud 附可用 preset 名清单
- `resume_study_from` 给定而 `inherit_platform_job_id` 缺 → 提交侧静态门 fail-loud 提示两参
- `inherit_platform_job_id` 指向不属于 E 的 job / outputs 无 `{id}.zip` → fail-loud
- `--kwargs-json` 载荷 >128KiB → provider 预检 fail-loud 提示拆分
- local 模式带 platform_experiment_id/inherit → 忽略且恒 `finished`+`dag_result`（模式专属参数原则，先例 headers）

## S4 在途观察：`get_platform_job_logs`（🔴 blockedBy #46）

- J1 轮询：返回 `{job_id, status, terminal, tail_lines, logs}`；未终态 `terminal=false` 继续；`succeeded` → `terminal=true` 停轮
- `tail_lines` 边界：0 / 负 / 10001 → 本地 fail-loud（不发出 HTTP；上游 `<=0→500` 是其容错非我方语义）

## S5 拉取与本地查询（🟢 实现后即测；download-url 属 /experiments 下 files 面——CLI scope 放行依据同 S2，未逐路实测，复测 403 则随 #46 转 🔴）

- `download_platform_job_artifacts(job_id=J1, platform_experiment_id=E, target_dir=<已存在>)` → 解压落位 `target_dir/{J1}/`（zip 根 = workspace 根）；返回 `{job_id, platform_experiment_id, local_task_dir, file_count, mlflow_db:<path>}`；zip 本体即弃（target_dir 无 .zip 残留）
- zip 内容断言（hpo 形态）：`mlflow.db` + `.mlruns/` 树 + `.mflowy/study/<run_id>.db` 活文件 + run 工件 `study.db`
- 重下载 → 原子替换（内容最新、无旧文件残留、幂等）
- `get_run(tracking_uri="sqlite:///<target_dir>/{J1}/mlflow.db")` → 查到 J1 模型步 run，**取 run_id 供 S6**；返回工件路径已从 `file:///user-data/workspace/...` 改写为 `<target_dir>/{J1}/...` 本地路径（db 所在目录推导、单参式；不匹配前缀原样透传）
- `get_experiment`/`get_run` 不传 tracking_uri → 机器默认库，既有行为回归不变

**负向**：target_dir 不存在 → fail-loud；MCP 通道相对 target_dir → 拒收。

## S6 续跑链：继承 × resume × 跨 Job 寻址（🔴 blockedBy #46）

三个 job 依次落在**同一平台实验 E**（同实验宿主多 Job）：

- **J2 = hpo 续跑**：`inherit_platform_job_id=J1` + `resume_study_from=<S5 取得的 run_id>` → `succeeded`；断言：
  - J2 zip 内 study 库 trial 计数 = J1 累计 + J2 新增（本地 sqlite 读 `.mflowy/study/<J2run>.db`）——增量续跑实证
  - J2 zip 的 mlflow.db 含 J1 runs（继承解包滚雪球）
- **J3 = modeling 训练**（#161 载荷）：`inherit_platform_job_id=J2` → `succeeded`；zip 追加 `X_test/y_test/y_pred.parquet` + 评估图 + `.mlruns/models/m-<uuid>/`（logged model 本体，#155 双形态）
- **J4 = explanation**：`model=<J3 模型步 run_id>` + `inherit_platform_job_id=J3` + `platform_experiment_id=E` → `succeeded`——**跨 Job 寻址**硬前提实证（前序 run 只在该实验继承链的 mlflow.db 里可寻址）

**负向**：inherit 指向无 zip 的 job → fail-loud（S3 已覆盖）；「坏 zip fail-loud 带同实验候选 zip 清单」E2E 不可编排（outputs 由平台管理），归 jobwrap 单测覆盖。

## S7 观察面收口（🟢 实现后即测；纯本地 mlflow 面，素材依赖 S6 产物）

- download J3 → `get_experiment(tracking_uri=J3 db)` → 实验内 runs 数 = 继承链累计（J1+J2+J3）
- J4 的 explanation run 工件（run 形态）与 J3 logged model（models/ 形态）路径**双形态均改写**为本地路径
- 改写常量命中判定：不匹配 `/user-data/workspace` 前缀的 URI 原样透传（负向以 Default 实验路径探针）

## S8 横切负向（🟢 实现后即测）

- 平台不可达：`MFLOWY_BASE_URL` 指向死端口 → `PlatformApiError`，message 含 base_url 与 NodePort 可达性提示
- 遥测门：`MFLOWY_TELEMETRY=off` 下全链零 elicitation
- 401 强刷单次重放 / 坏 zip 候选清单 / 128KiB 上限边界：单测覆盖项，E2E 不重复编排

## 汇总：PR 切片 → 必过阶段（建议，实际切片由实现侧定）

| 实现切片 | 必过 | 上游 |
| --- | --- | --- |
| ① 镜像与依赖基座（Dockerfile 合入 + httpx2/pydantic 进 model 默认依赖） | S0 S1 | 无 |
| ② platform_client + 错误族（单测为主） | S0 S1 S8 | 无 |
| ③ ensure_platform_experiment（mcp 两站注册） | S2 | 无（复测 upload-url scope） |
| ④ RemoteJobProvider + `mflowy-job wrap` + JobOutcome 协议/compute 参数面 | S3 S4 S5 | #46 |
| ⑤ download 工具 + 观察面单参改造 | S5 S7 | 无 |
| ⑥ resume_study_from 机制 + 系列收尾 | 全链 S1–S8 | #46（收尾须 #46 已扩表） |

**通过判据**：所涉阶段断言全绿。每轮留档（附 PR 描述）：分支 sha、镜像 tar 路径、experiment id、全部 job id、每 job 日志尾、zip 清单（`unzip -l`）、驱动脚本输出。

## 附录：#161 基座载荷（已实证）

`payload.json`（modeling 工具，diabetes 最小链 load→clean→x_y→cv→XGB，`n_trials: 2`）：

```json
{"modeling_steps": "- load__python:\n    name: 加载\n    params:\n      source: /user-data/uploads/diabetes.py:load\n- clean__python:\n    name: 删除空行与空列\n    params:\n      source: /user-data/uploads/diabetes.py:clean\n- x_y:\n    name: X/y 拆分\n    params:\n      target: target\n- cv__split:\n    name: 数据集划分\n    params:\n      test_size: 0.2\n- model__XGB:\n    name: XGBoost\n    params:\n      n_estimators: 30\n      n_trials: 2\n      scoring: mae\n", "name": "base-verify", "desc": "#161 远程任务基座验证"}
```

- S1 的 task.sh 复用 #161 留档（`/tmp/mflowy-base-verify/task.sh`：ENV 回显 / 词表 / seam pin / `mflowy modeling` / 手工 zip / 只读探针）；实现后 wrapper 取代其 [5][6] 段，S1 保留基座版作平台回归。
- hyperparameters_search / explanation 载荷按各工具 schema 构造（`tools/list` 即文档），首个实现 PR 定稿后回填此处。
