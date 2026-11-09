# 上游 #44 落地状态核实（mflowy-platform）

> 调研票：ifoodsci-ai/mflowy#154 · 上游 spec：lincode7/mflowy-platform#44（Job 挂载拆三 — uploads RO / outputs PVC / workspace emptyDir scratch + preset `workspace_size_mi`）
> 核实日期：2026-11-09 · 证据基线：本地 /home/lin/mflowy-platform `dev` == `origin/dev`（`21b6e01f`，fetch 后比对一致）；salvage/pull/{1,2,3} 旧远端分支亦无相关提交。

## 一句话结论

**#44 尚未落地（今日 2026-11-09 刚建 spec，OPEN / ready-for-agent / 0 评论 / 0 PR / 0 提交），语义契约不可作为已交付 API 消费——mflowy 侧 RemoteJobProvider 适配「需等待」**；可先行按 spec 契约做纸面设计（字段名/语义已定稿），但联调与验收断言必须等上游实现合入。

## 四点核实

### (1) 关联 PR 与挂载现状 —— 无 PR，job.go 仍是旧单 subPath 挂载

- PR 检索（`gh pr list --repo lincode7/mflowy-platform --state all`）：`44` / `挂载` / `emptyDir` / `workspace_size` 四组关键词均无关联 PR；命中的仅 #43（deer-flow 升级，无关）与 #29（v2 Job 全链，11-01 合入，属 #44 之前的旧实现）。
- git log（`--all` 含全部远端与 salvage refs，grep `workspace_size|emptydir|挂载拆|uploads.*ro`）：零命中。最近的 job 系提交全部属 PR #29（`4ca60467` NewTrainingJob v2「实验工作区挂载」、`5b2aabf0` JobManager 等）。
- 代码现状 `apps/server/utils/infra/k8s/job.go`（361 行）：**旧单挂载**——单一 PVC 卷（`spec.UserDataPVC`）以 `SubPath: files/{org}/{exp}/user-data` 挂到 `/user-data`（L224/L266-276），uploads/workspace/outputs 三目录同居一个 RWX 挂载点下（L23-24 注释明示）。无 emptyDir、无 uploads 只读、无独立 outputs 卷。
- 全仓 `git grep -i "workspace_size|emptyDir" origin/dev`（排除 apps/services/mflowy 子仓）：仅命中 csi-s3 provisioner 模板与一个无关测试函数名 `TestService_ListFiles_EmptyDirPruning`。

**结论**：#44 纯 spec、零实现；当前挂载语义 = 单 PVC 单 subPath、三目录同挂载点同读写模式——与 #44 目标语义（三挂载/RO/PVC 配额）完全不符。

### (2) JobPreset / ListJobPresets —— 无 workspace_size_mi 字段

- `apps/server/apis/job.proto` `JobPreset`（L38-52）字段止于 `workdir=10` + `labels=11` + `annotations=12` + `created_at=13`；资源面仅 `cpu_millis/memory_mi/gpu_count/active_deadline_seconds/image/ttl_seconds_after_finished`。
- `ListJobPresetsResponse`（L126-129）= `repeated JobPreset presets`，随消息体自然不含 `workspace_size_mi`。`CreateJobPresetRequest`/`NewJobPresetVersionRequest` 同样无此字段（L132-169）。
- proto/go/sql 全源 grep `workspace_size`：零命中（DB 面 `job_presets` schema 亦无，见 #29 提交 `9e63bef1`）。

**结论**：preset 的 workspace 配额面尚未存在，字段名 `workspace_size_mi` 目前只活在 spec 里。

### (3) CreateJob 输入面 —— 确认仍为四字段、无 env 面（现状即 #44 前旧契约）

- `job.proto` `CreateJobRequest`（L60-65）逐字：`cmd=1 / args=2 (repeated string) / experiment_id=3 / preset_id=4`，注释明示 "every field required (no implicit defaults)"；无 `env`、无挂载相关字段。
- 与 #44 spec 的「CreateJob 零改动」预期**方向一致但属存量态**——即当前无 env 面是旧实现如此，非 #44 落地后的确认；#44 合入时此面预期不变。

**结论**：CreateJob 输入面今天就是 `cmd/args/experiment_id/preset_id` 四字段，可据此固化 mflowy 侧 build command 设计；但仍需在上游合入后回归确认未被 spec 外字段污染。

### (4) acceptance runbook —— 无「全新实验首 job」场景、无 200→404 翻转断言

- `apps/server/experiment-service/docs/acceptance-job-runbook.md`（spec #22 v2 时代产物，PR #29 `b9a0eae9` 落盘）：八步验收流程**前置假设 uploads 已上传物化**（步骤 2 先传 task.sh/test.txt），全程无「全新实验、零上传、直接首 job」的场景分支。
- 唯一相邻事实是 L29 的目录物化注记（geesefs 只呈现有对象路径，workspace/outputs 无对象不出现），但无任何「workspace 精确名先 200 后 404（或反之）翻转」的断言步骤。

**结论**：#44 spec 要求的验收断言场景尚未写入 runbook——runbook 更新属 #44 实现工作的一部分，当前未发生。

## 对 mflowy 侧（地图 #153）的判定

- **需等待**：挂载拆三 + `workspace_size_mi` preset 面均为 spec-only，mflowy 的 RemoteJobProvider 无法对真实行为做任何依赖（尤其「workspace 为 emptyDir scratch、job 间不共享」的写路径语义，现状是 RWX 共享——两者行为相反）。
- 可并行推进的纸面工作：CreateJob 四字段输入面（第 3 点）与 preset 选择逻辑的字段名可先按 spec 定形；「workspace 不可作持久通道」的 mflowy 侧纪律（输出一律走 outputs）与 #44 目标语义一致，可提前写进设计。
- 回归触发器：上游出现引用 #44 的 PR 合入 origin/dev 后，重跑本清单四点即可判可用性。

## 证据索引

| 事实 | 证据 |
| --- | --- |
| #44 OPEN / 今日新建 / 0 评论 | `gh issue view 44 --repo lincode7/mflowy-platform`（createdAt 2026-11-09T11:22:21Z） |
| 无关联 PR | `gh pr list --state all` 四关键词检索（44/挂载/emptyDir/workspace_size） |
| 本地不落后 | `git -C /home/lin/mflowy-platform rev-parse dev origin/dev` 同为 `21b6e01f` |
| 旧单 subPath 挂载 | `apps/server/utils/infra/k8s/job.go` L23-25, L224, L266-276 |
| JobPreset 无 workspace_size_mi | `apps/server/apis/job.proto` L38-52 |
| CreateJob 四字段 | `apps/server/apis/job.proto` L56-65 |
| runbook 缺场景/断言 | `apps/server/experiment-service/docs/acceptance-job-runbook.md` 全文（步骤 2 前置上传、L29 物化注记） |
